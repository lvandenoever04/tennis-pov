import os
import sys
import subprocess
import cv2
import numpy as np
import torch

if len(sys.argv) < 5:
    sys.exit("Usage: python 12_track_ball_tracknet.py VIDEO FIRST LAST WEIGHTS\n"
             "   e.g. python 12_track_ball_tracknet.py data/centre1.mp4 0 845 ~/projects/TrackNet/model_best.pt")
path, first, last, weights = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), os.path.expanduser(sys.argv[4])
n = last - first + 1

# Use the TrackNet repo's own model definition and post-processing.
sys.path.insert(0, os.path.expanduser("~/projects/TrackNet"))
from model import BallTrackerNet    # type: ignore
from general import postprocess     # type: ignore

NET_W, NET_H = 640, 360     # the network works on frames shrunk to this size
OUTLIER = 150               # px: a detection this far from BOTH neighbours is treated as a mistake
MAX_FILL = 4                # frames: gaps up to this long are filled by joining the dots

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BallTrackerNet()
model.load_state_dict(torch.load(weights, map_location=device))
model.to(device).eval()

video = cv2.VideoCapture(path)
fps = video.get(cv2.CAP_PROP_FPS)
w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
scale = w / NET_W           # 1920 / 640 = 3: back from network size to full size

# The network needs the two frames before each one, so start reading 2 frames early (if there are any).
start = max(first - 2, 0)
video.set(cv2.CAP_PROP_POS_FRAMES, start)
small = []                  # recent frames, shrunk to network size
ball = np.full((n, 2), np.nan)
for f in range(start, last + 1):
    ok, frame = video.read()
    if not ok:
        break
    small.append(cv2.resize(frame, (NET_W, NET_H)))
    small = small[-3:]
    if f < first or len(small) < 3:
        continue
    # current, previous, one before that -> 9 channels, values 0-1, channels first, plus a batch dimension
    stacked = np.concatenate([small[2], small[1], small[0]], axis=2).astype(np.float32) / 255.0
    x_in = torch.from_numpy(np.rollaxis(stacked, 2, 0)[None]).to(device)
    with torch.no_grad():
        out = model(x_in).argmax(dim=1).cpu().numpy()
    x, y = postprocess(out, scale=scale)
    if x is not None:
        ball[f - first] = (x, y)
    if (f - first) % 100 == 0:
        print(f"  frame {f}")

# Remove isolated mistakes: a detection far from both its neighbours.
for k in range(1, n - 1):
    if np.isnan(ball[k, 0]) or np.isnan(ball[k - 1, 0]) or np.isnan(ball[k + 1, 0]):
        continue
    if np.hypot(*(ball[k] - ball[k - 1])) > OUTLIER and np.hypot(*(ball[k] - ball[k + 1])) > OUTLIER:
        ball[k] = np.nan

source = np.where(np.isnan(ball[:, 0]), 0, 1)      # 0 = none, 1 = detected, 2 = filled in


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


for a, b in runs(source == 0):                       # fill short gaps between two detections
    if a > 0 and b < n - 1 and b - a + 1 <= MAX_FILL:
        for k in range(a, b + 1):
            t = (k - a + 1) / (b - a + 2)
            ball[k] = (1 - t) * ball[a - 1] + t * ball[b + 1]
            source[k] = 2

gaps = sorted(runs(source == 0), key=lambda g: g[0] - g[1])
print(f"Ball found in {np.sum(source == 1)} frames, filled in {np.sum(source == 2)}, "
      f"missing in {np.sum(source == 0)} of {n}")
print("Longest gaps: " + ", ".join(f"{first + a}-{first + b} ({b - a + 1})" for a, b in gaps[:6]))
np.savez("data/ball2d.npz", range=np.array([first, last]), position=ball, source=source, fps=fps)
print("Saved data/ball2d.npz")

# Video: ball circled, with a short trail
video.set(cv2.CAP_PROP_POS_FRAMES, first)
ok, frame = video.read()
h, w = frame.shape[:2]
writer = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                           "-pix_fmt", "yuv420p", "data/ball2d.mp4"], stdin=subprocess.PIPE)
for k in range(n):
    if k > 0:
        ok, frame = video.read()
    for j in range(max(0, k - 12), k):
        if source[j] and source[j + 1]:
            cv2.line(frame, tuple(ball[j].astype(int)), tuple(ball[j + 1].astype(int)), (0, 255, 255), 2, cv2.LINE_AA)
    if source[k]:
        colour = (0, 0, 255) if source[k] == 1 else (255, 0, 255)
        cv2.circle(frame, tuple(ball[k].astype(int)), 14, colour, 2, cv2.LINE_AA)
    label = {0: "no ball", 1: "ball", 2: "ball (filled in)"}[source[k]]
    text = f"frame {first + k}  {label}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
    x0 = (w - tw) // 2                                           # centred horizontally
    cv2.rectangle(frame, (x0 - 12, 18), (x0 + tw + 12, 30 + th + 12), (0, 0, 0), -1)
    cv2.putText(frame, text, (x0, 30 + th), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    writer.stdin.write(frame.tobytes())
writer.stdin.close()
writer.wait()
print("Saved data/ball2d.mp4")