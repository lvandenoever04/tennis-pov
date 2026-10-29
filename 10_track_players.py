import os
import sys
import time
import subprocess
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import court
from geometry import Camera

if len(sys.argv) < 4:
    sys.exit("Usage: python 10_track_players.py VIDEO FIRST_FRAME LAST_FRAME   e.g. ... data/centre1.mp4 0 845")
path, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
n = last - first + 1

DETECTIONS = "data/detections.npz"   # cache: delete this file to run the detector again
LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE = 11, 12, 15, 16    # positions in the 17-point list

# who is who
MAX_JUMP = 1.0            # m: furthest a player can be from their last position, one frame later
EXTRA_PER_MISSED = 0.35   # m: the gate widens by this for every frame the player has been missing
MIN_HEIGHT = 1.1          # m: shorter detections (crouched ball kids) are penalised

# measuring and smoothing
ANKLE_HEIGHT = 0.09       # m: ankle joints sit about this far above the grass
ANKLE_PIXEL_NOISE = 4.0   # px: how much an ankle keypoint wobbles frame to frame
HIP_PIXEL_NOISE = 8.0     # px: hips are vaguer, and their height changes as the player crouches
MIN_KEYPOINT_CONF = 0.5   # keypoints the network is less sure about than this are ignored
EDGE = 3                  # px: a box this close to the bottom of the picture has its feet cut off
ACCELERATION = 6.0        # m/s^2: how hard a player typically speeds up / slows down / turns

cam = Camera()
video = cv2.VideoCapture(path)
fps = video.get(cv2.CAP_PROP_FPS)
image_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ---- 1. Detect everyone in every frame (slow, so the result is cached) ----
cached = os.path.exists(DETECTIONS) and tuple(np.load(DETECTIONS)["range"]) == (first, last)
if cached:
    d = np.load(DETECTIONS)
    print(f"Loaded cached detections from {DETECTIONS}")
else:
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO("yolo11x-pose.pt")
    rows = {"frame": [], "box": [], "conf": [], "keypoints": [], "keypoint_conf": []}
    video.set(cv2.CAP_PROP_POS_FRAMES, first)
    start = time.time()
    for k in range(n):
        ok, frame = video.read()
        r = model.predict(frame, imgsz=1920, conf=0.2, device=device, verbose=False)[0]
        kc = r.keypoints.conf
        kc = kc.cpu().numpy() if kc is not None else np.ones(r.keypoints.xy.shape[:2])
        rows["frame"] += [first + k] * len(r.boxes)
        rows["box"] += list(r.boxes.xyxy.cpu().numpy())
        rows["conf"] += list(r.boxes.conf.cpu().numpy())
        rows["keypoints"] += list(r.keypoints.xy.cpu().numpy())
        rows["keypoint_conf"] += list(kc)
        if k % 100 == 0:
            print(f"  detecting: frame {first + k} ({(k + 1) / (time.time() - start):.1f} frames/s)")
    d = {key: np.array(v) for key, v in rows.items()}
    np.savez(DETECTIONS, range=np.array([first, last]), **d)
    print(f"Detected {len(d['frame'])} people in {n} frames ({time.time() - start:.0f} s); saved {DETECTIONS}")

det_frame, boxes, kps, kp_conf = d["frame"], d["box"], d["keypoints"], d["keypoint_conf"]

# ---- 2. Rough court position and height of every detection (only used to decide who is who) ----
ground = np.array([cam.to_court((b[0] + b[2]) / 2, b[3]) for b in boxes])
height = np.array([cam.height_above_ground((b[0] + b[2]) / 2, b[1], *g) for b, g in zip(boxes, ground)])
in_frame = [np.nonzero(det_frame == first + k)[0] for k in range(n)]

# ---- 3. Who is who: follow each player from frame to frame ----
tracks = {}
for name, side in (("near", -1), ("far", +1)):
    chosen = np.full(n, -1)
    last_pos, missed = None, 0
    for k in range(n):
        options = [i for i in in_frame[k]
                   if side * ground[i, 1] > -0.5 and abs(ground[i, 0]) < 9 and abs(ground[i, 1]) < 19]
        if last_pos is None:
            options = [i for i in options if height[i] > MIN_HEIGHT and abs(ground[i, 0]) < 7]
            if not options:
                continue
            best = min(options, key=lambda i: np.hypot(ground[i, 0], ground[i, 1] - side * court.HALF_LENGTH))
        else:
            gate = MAX_JUMP + EXTRA_PER_MISSED * missed
            options = [i for i in options if np.hypot(*(ground[i] - last_pos)) <= gate]
            if not options:
                missed += 1
                continue
            best = min(options, key=lambda i: np.hypot(*(ground[i] - last_pos)) + (0.5 if height[i] < MIN_HEIGHT else 0))
        chosen[k] = best
        last_pos, missed = ground[best], 0
    tracks[name] = chosen


# ---- 4. Measure: body points at known heights -> court positions, each with an uncertainty ----
def ray_to_height(u, v, z):
    """Follow the camera ray through pixel (u, v) until it is z metres above the grass; return (x, y)."""
    ray = cam.R.T @ np.array([(u - cam.cx) / cam.f, (v - cam.cy) / cam.f, 1.0])
    t = (z - cam.C[2]) / ray[2]
    return (cam.C + t * ray)[:2]


def uncertainty(u, v, z, pixel_noise):
    """2x2 covariance of the court position: how much it moves per pixel of error, in each direction."""
    p = ray_to_height(u, v, z)
    J = np.stack([ray_to_height(u + 1, v, z) - p, ray_to_height(u, v + 1, z) - p], axis=1)
    return pixel_noise ** 2 * J @ J.T


def visible(i, a, b):
    return kp_conf[i, [a, b]].min() > MIN_KEYPOINT_CONF


def midpoint(i, a, b):
    return kps[i, [a, b]].mean(axis=0)


def kalman_smooth(measured, covariance, dt, acceleration):
    """Most likely path given noisy measurements (nan = none) and a limit on how hard a person accelerates.
    State: x, y, x-speed, y-speed. Returns states (n, 4) and their covariances (n, 4, 4)."""
    count = len(measured)
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt                               # position += speed * dt
    G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
    Q = acceleration ** 2 * G @ G.T                      # how much an unknown push can change things per frame
    Hm = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])      # we measure position, not speed
    start = np.nonzero(~np.isnan(measured[:, 0]))[0][0]
    x = np.array([*measured[start], 0.0, 0.0])
    P = np.diag([1.0, 1.0, 25.0, 25.0])
    predicted_x, predicted_P, filtered_x, filtered_P = [], [], [], []
    for k in range(count):                               # forwards: predict, then correct with the measurement
        if k > 0:
            x, P = F @ x, F @ P @ F.T + Q
        predicted_x.append(x)
        predicted_P.append(P)
        if not np.isnan(measured[k, 0]):
            S = Hm @ P @ Hm.T + covariance[k]
            K = P @ Hm.T @ np.linalg.inv(S)              # how much to trust this measurement vs the prediction
            x = x + K @ (measured[k] - Hm @ x)
            P = (np.eye(4) - K @ Hm) @ P
        filtered_x.append(x)
        filtered_P.append(P)
    xs, Ps = np.array(filtered_x), np.array(filtered_P)
    for k in range(count - 2, -1, -1):                   # backwards: let later frames improve earlier ones
        A = filtered_P[k] @ F.T @ np.linalg.inv(predicted_P[k + 1])
        xs[k] = filtered_x[k] + A @ (xs[k + 1] - predicted_x[k + 1])
        Ps[k] = filtered_P[k] + A @ (Ps[k + 1] - predicted_P[k + 1]) @ A.T
    return xs, Ps


def runs(mask):
    out, k = [], 0
    while k < len(mask):
        if mask[k]:
            j = k
            while j < len(mask) and mask[j]:
                j += 1
            out.append((k, j - 1))
            k = j
        else:
            k += 1
    return out


NONE, FEET, HIPS = 0, 1, 2
results = {}
for name, chosen in tracks.items():
    measured = np.full((n, 2), np.nan)
    covariance = np.full((n, 2, 2), np.nan)
    method = np.zeros(n, int)
    hip_heights = []
    for k in range(n):                                   # first choice: ankles, if the feet are in the picture
        i = chosen[k]
        if i >= 0 and boxes[i, 3] < image_height - EDGE and visible(i, LEFT_ANKLE, RIGHT_ANKLE):
            u, v = midpoint(i, LEFT_ANKLE, RIGHT_ANKLE)
            measured[k] = ray_to_height(u, v, ANKLE_HEIGHT)
            covariance[k] = uncertainty(u, v, ANKLE_HEIGHT, ANKLE_PIXEL_NOISE)
            method[k] = FEET
            if visible(i, LEFT_HIP, RIGHT_HIP):          # learn this player's hip height while we can see both
                hip_heights.append(cam.height_above_ground(*midpoint(i, LEFT_HIP, RIGHT_HIP), *measured[k]))
    hip_height = float(np.median(hip_heights))
    for k in range(n):                                   # second choice: hips
        i = chosen[k]
        if i >= 0 and method[k] == NONE and visible(i, LEFT_HIP, RIGHT_HIP):
            u, v = midpoint(i, LEFT_HIP, RIGHT_HIP)
            measured[k] = ray_to_height(u, v, hip_height)
            covariance[k] = uncertainty(u, v, hip_height, HIP_PIXEL_NOISE)
            method[k] = HIPS

    states, state_cov = kalman_smooth(measured, covariance, 1 / fps, ACCELERATION)
    position, velocity = states[:, :2], states[:, 2:]
    speed = np.hypot(velocity[:, 0], velocity[:, 1])
    spread = np.sqrt(np.stack([state_cov[:, 0, 0], state_cov[:, 1, 1]], axis=1))   # +/- metres
    gaps = runs(method == NONE)
    longest = max(gaps, key=lambda g: g[1] - g[0]) if gaps else None
    print(f"\n{name} player (hips {hip_height:.2f} m above the grass):")
    print(f"  measured from feet {np.sum(method == FEET)}, from hips {np.sum(method == HIPS)}, "
          f"not at all {np.sum(method == NONE)} of {n} frames")
    print(f"  top speed {speed.max():.1f} m/s (frame {first + int(np.argmax(speed))})")
    if longest:
        a, b = longest
        print(f"  longest stretch with no measurement: frames {first + a}-{first + b}, "
              f"uncertainty there up to +/-{spread[a:b + 1].max():.2f} m")
    results[name] = {"position": position, "speed": speed, "spread": spread, "method": method, "chosen": chosen}

save = {"range": np.array([first, last]), "fps": fps}
for name, r in results.items():
    c = r["chosen"]
    save[f"{name}_position"] = r["position"]      # smoothed court position (m), every frame
    save[f"{name}_speed"] = r["speed"]            # m/s
    save[f"{name}_uncertainty"] = r["spread"]     # +/- m in x and y
    save[f"{name}_method"] = r["method"]          # 0 = estimated, 1 = from feet, 2 = from hips
    save[f"{name}_box"] = np.where(c[:, None] >= 0, boxes[np.maximum(c, 0)], np.nan)
    save[f"{name}_keypoints"] = np.where(c[:, None, None] >= 0, kps[np.maximum(c, 0)], np.nan)
    save[f"{name}_keypoint_conf"] = np.where(c[:, None] >= 0, kp_conf[np.maximum(c, 0)], np.nan)
np.savez("data/players.npz", **save)
print("\nSaved data/players.npz")

# ---- 5. Top-down picture: measured parts in colour, estimated parts in grey ----
SCALE, W, H = 20, 440, 800
def to_map(x, y):
    return int(W / 2 + x * SCALE), int(H / 2 - y * SCALE)
topdown = np.full((H, W, 3), (60, 120, 60), np.uint8)
for (x0, y0), (x1, y1) in court.LINES:
    cv2.line(topdown, to_map(x0, y0), to_map(x1, y1), (255, 255, 255), 1, cv2.LINE_AA)
cv2.line(topdown, to_map(-6, 0), to_map(6, 0), (30, 30, 30), 2)
bottom_of_picture = cam.to_court(cam.cx, image_height)[1]
cv2.line(topdown, to_map(-11, bottom_of_picture), to_map(11, bottom_of_picture), (0, 200, 255), 1)
cv2.putText(topdown, "bottom of picture", to_map(-10.5, bottom_of_picture - 0.6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
for name, colour in (("near", (60, 60, 230)), ("far", (230, 150, 50))):
    r = results[name]
    pts = [to_map(x, y) for x, y in r["position"]]
    for k in range(1, n):
        c = colour if r["method"][k] != NONE else (170, 170, 170)
        cv2.line(topdown, pts[k - 1], pts[k], c, 2, cv2.LINE_AA)
    cv2.putText(topdown, name, (pts[0][0] + 8, pts[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
cv2.imwrite("data/players_topdown.png", topdown)
print("Saved data/players_topdown.png")

# ---- 6. Video: each player labelled with speed and how they were measured ----
video.set(cv2.CAP_PROP_POS_FRAMES, first)
ok, frame = video.read()
h, w = frame.shape[:2]
writer = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                           "-pix_fmt", "yuv420p", "data/players.mp4"], stdin=subprocess.PIPE)
how = {NONE: "estimated", FEET: "feet", HIPS: "hips"}
for k in range(n):
    if k > 0:
        ok, frame = video.read()
    for name, colour in (("near", (60, 60, 230)), ("far", (230, 150, 50))):
        r = results[name]
        gx, gy = r["position"][k]
        (u, v), = cam.project([[gx, gy, 0]])
        cv2.ellipse(frame, (int(u), int(v)), (30, 10), 0, 0, 360, colour, 2, cv2.LINE_AA)
        if r["chosen"][k] >= 0:
            x0, y0, x1, y1 = boxes[r["chosen"][k]].astype(int)
            cv2.rectangle(frame, (x0, y0), (x1, y1), colour, 2)
        label = f"{name} {r['speed'][k]:.1f} m/s ({how[r['method'][k]]})"
        tx, ty = int(np.clip(u - 80, 10, w - 330)), int(np.clip(v + 35, 30, h - 40))
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    cv2.putText(frame, f"frame {first + k}", (20, 1050), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    writer.stdin.write(frame.tobytes())
writer.stdin.close()
writer.wait()
print("Saved data/players.mp4")