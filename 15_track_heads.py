import sys
import subprocess
import cv2
import numpy as np
from scipy.signal import savgol_filter
from geometry import Camera

if len(sys.argv) < 2:
    sys.exit("Usage: python 15_track_heads.py VIDEO   e.g. python 15_track_heads.py data/centre25.mp4")
path = sys.argv[1]

EYES_EARS = [1, 2, 3, 4]       # keypoints: left eye, right eye, left ear, right ear (the nose sticks out in front)
SHOULDERS = [5, 6]
MIN_CONF = 0.5                 # keypoints the network is less sure about than this are ignored
SMOOTH_FRAMES = 7              # smoothing window (odd)
HEIGHTS = {"near": ("Alcaraz", 1.83), "far": ("Fognini", 1.78)}   # m, for the sanity check only
EYE_BELOW_TOP = 0.12           # m: eyes are about this far below the top of the head

cam = Camera()
p = np.load("data/players.npz")
first, last = (int(v) for v in p["range"])
fps = float(p["fps"])
n = last - first + 1


def ray(u, v):
    """Direction (in court coordinates) of the camera ray through pixel (u, v)."""
    return cam.R.T @ np.array([(u - cam.cx) / cam.f, (v - cam.cy) / cam.f, 1.0])


def point_above(u, v, gx, gy):
    """The point on the ray through (u, v) that passes most directly above the ground point (gx, gy)."""
    d = ray(u, v)
    t = ((gx - cam.C[0]) * d[0] + (gy - cam.C[1]) * d[1]) / (d[0] ** 2 + d[1] ** 2)
    return cam.C + t * d


FACE, FROM_SHOULDERS, ESTIMATED = 1, 2, 0
results = {}
for name in ("near", "far"):
    body = p[f"{name}_position"]                   # smoothed court position of the body (feet), every frame
    kps, conf = p[f"{name}_keypoints"], p[f"{name}_keypoint_conf"]
    head = np.full((n, 3), np.nan)
    source = np.zeros(n, int)
    shoulder = np.full((n, 3), np.nan)

    for k in range(n):
        if np.isnan(conf[k]).all():                # player not detected in this frame
            continue
        gx, gy = body[k]
        good = [i for i in EYES_EARS if conf[k, i] > MIN_CONF]
        if good:
            head[k] = point_above(*kps[k, good].mean(axis=0), gx, gy)
            source[k] = FACE
        if conf[k, SHOULDERS].min() > MIN_CONF:
            shoulder[k] = point_above(*kps[k, SHOULDERS].mean(axis=0), gx, gy)

    # shoulder-to-eye height, measured from frames where both were seen (not assumed)
    both = (source == FACE) & ~np.isnan(shoulder[:, 0])
    rise = float(np.median(head[both, 2] - shoulder[both, 2])) if both.sum() > 10 else 0.23
    use = (source == ESTIMATED) & ~np.isnan(shoulder[:, 0])
    head[use] = shoulder[use] + [0, 0, rise]
    source[use] = FROM_SHOULDERS

    # fill the rest (player off-screen): follow the body, at the interpolated head height and lean
    seen = source != ESTIMATED
    t = np.arange(n)
    lean = head[:, :2] - body                      # how far the head is from above the feet
    for d in (0, 1):
        lean[:, d] = np.interp(t, t[seen], lean[seen, d])
    height = np.interp(t, t[seen], head[seen, 2])

    # smooth, so it moves like a head (the body path is already smooth from script 10)
    lean = savgol_filter(lean, SMOOTH_FRAMES, 2, axis=0)
    height = savgol_filter(height, SMOOTH_FRAMES, 2)
    head = np.c_[body + lean, height]
    results[name] = (head, source)

    # ---- report ----
    who, tall = HEIGHTS[name]
    speed = np.linalg.norm(np.diff(head, axis=0), axis=1) * fps
    offset = np.hypot(*lean.T)
    print(f"\n{name} player ({who}, {tall} m tall: eyes about {tall - EYE_BELOW_TOP:.2f} m up when standing straight):")
    print(f"  head from eyes/ears {np.sum(source == FACE)}, from shoulders {np.sum(source == FROM_SHOULDERS)}, "
          f"estimated {np.sum(source == ESTIMATED)} of {n} frames (shoulder to eyes: {rise:.2f} m)")
    print(f"  eye height: typical {np.median(height):.2f} m, highest {np.percentile(height, 98):.2f} m, "
          f"lowest {height.min():.2f} m (frame {first + int(np.argmin(height))})")
    print(f"  head vs feet: typically {np.median(offset):.2f} m apart sideways, at most {offset.max():.2f} m")
    print(f"  head top speed {speed.max():.1f} m/s (frame {first + int(np.argmax(speed))})")

np.savez("data/heads.npz", range=p["range"], fps=fps,
         **{f"{name}_head": h for name, (h, s) in results.items()},
         **{f"{name}_head_source": s for name, (h, s) in results.items()})
print("\nSaved data/heads.npz")

# ---- picture: eye height over the rally ----
W, H, top_m, bottom_m = max(n, 700), 320, 2.1, 0.6
graph = np.full((H, W, 3), 30, np.uint8)
def gy_(z):
    return int((top_m - z) / (top_m - bottom_m) * (H - 20) + 10)
for z in (0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0):
    cv2.line(graph, (0, gy_(z)), (W, gy_(z)), (70, 70, 70), 1)
    cv2.putText(graph, f"{z:.1f} m", (4, gy_(z) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
for name, colour in (("near", (60, 60, 230)), ("far", (230, 150, 50))):
    head, source = results[name]
    eye = HEIGHTS[name][1] - EYE_BELOW_TOP
    cv2.line(graph, (0, gy_(eye)), (W, gy_(eye)), colour, 1, cv2.LINE_AA)        # standing eye height
    pts = np.array([(int(k * W / n), gy_(z)) for k, z in enumerate(head[:, 2])], np.int32)
    cv2.polylines(graph, [pts], False, colour, 2, cv2.LINE_AA)
cv2.putText(graph, "eye height over the rally (red: near, blue: far; thin lines: standing eye height)",
            (60, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
cv2.imwrite("data/head_height.png", graph)
print("Saved data/head_height.png")

# ---- video: head (circle) with a line straight down to the ground below it ----
video = cv2.VideoCapture(path)
video.set(cv2.CAP_PROP_POS_FRAMES, first)
w, h = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                           "-pix_fmt", "yuv420p", "data/heads.mp4"], stdin=subprocess.PIPE)
how = {FACE: "eyes/ears", FROM_SHOULDERS: "shoulders", ESTIMATED: "estimated"}
for k in range(n):
    ok, frame = video.read()
    if not ok:
        break
    for name, colour in (("near", (60, 60, 230)), ("far", (230, 150, 50))):
        head, source = results[name]
        (hu, hv), (fu, fv) = cam.project([head[k], [head[k][0], head[k][1], 0.0]])
        cv2.line(frame, (int(hu), int(hv)), (int(fu), int(fv)), colour, 1, cv2.LINE_AA)
        cv2.circle(frame, (int(hu), int(hv)), 7, colour, 2, cv2.LINE_AA)
        label = f"eyes {head[k][2]:.2f} m ({how[source[k]]})"
        cv2.putText(frame, label, (int(hu) + 12, int(hv) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame, label, (int(hu) + 12, int(hv) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    text = f"frame {first + k}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
    x0 = (w - tw) // 2
    cv2.rectangle(frame, (x0 - 12, 18), (x0 + tw + 12, 30 + th + 12), (0, 0, 0), -1)
    cv2.putText(frame, text, (x0, 30 + th), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    writer.stdin.write(frame.tobytes())
writer.stdin.close()
writer.wait()
print("Saved data/heads.mp4")