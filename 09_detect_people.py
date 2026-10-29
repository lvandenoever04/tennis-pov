import sys
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

if len(sys.argv) < 3:
    sys.exit("Usage: python 09_detect_people.py VIDEO FRAME   e.g. python 09_detect_people.py data/centre1.mp4 400")
path, frame_number = sys.argv[1], int(sys.argv[2])

# ---- GPU check ----
if torch.cuda.is_available():
    device = 0
    print(f"PyTorch {torch.__version__}, using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print(f"PyTorch {torch.__version__}: NO GPU FOUND, running on the CPU (slow)")

# ---- camera from scripts 7 and 8 ----
cam = np.load("data/camera.npz")
f, R, C, cx, cy = float(cam["f"]), cam["R"], cam["C"], float(cam["cx"]), float(cam["cy"])
H = np.load("data/court_track.npz")["homographies"][0]
H_inverse = np.linalg.inv(H)          # the homography maps court -> image; its inverse maps image -> court


def to_court(u, v):
    """A pixel on the ground -> court position (x across, y along) in metres."""
    p = H_inverse @ np.array([u, v, 1.0])
    return p[:2] / p[2]


def height_above_ground(u, v, gx, gy):
    """How high above the ground point (gx, gy) the pixel (u, v) is, in metres.
    The pixel defines a ray from the camera; we find where that ray passes directly above (gx, gy)."""
    ray = R.T @ np.array([(u - cx) / f, (v - cy) / f, 1.0])            # direction in court coordinates
    t = ((gx - C[0]) * ray[0] + (gy - C[1]) * ray[1]) / (ray[0] ** 2 + ray[1] ** 2)
    return C[2] + t * ray[2]


# ---- detect ----
video = cv2.VideoCapture(path)
video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ok, frame = video.read()

model = YOLO("yolo11x-pose.pt")       # the largest, most accurate pose model; downloads ~120 MB the first time
start = time.time()
# imgsz=1920: analyse at full resolution. The default (640) would shrink the far player to ~45 px tall.
result = model.predict(frame, imgsz=1920, conf=0.2, device=device, verbose=False)[0]
print(f"Detection took {time.time() - start:.2f} s (the first run is slower while the GPU warms up)")

boxes = result.boxes.xyxy.cpu().numpy()          # (n, 4): left, top, right, bottom in pixels
confidence = result.boxes.conf.cpu().numpy()     # (n,): how sure the network is, 0-1
keypoints = result.keypoints.xy.cpu().numpy()    # (n, 17, 2): pixel position of each body point
LEFT_ANKLE, RIGHT_ANKLE = 15, 16                 # positions in the standard 17-point list

print(f"\nFound {len(boxes)} people:")
print("  #  conf   feet: x across, y along (m)   ankles say     height (m)")
picture = result.plot(labels=False)                          # YOLO's own drawing: boxes + skeletons
for i, ((x0, y0, x1, y1), c) in enumerate(zip(boxes, confidence)):
    ankles = keypoints[i, [LEFT_ANKLE, RIGHT_ANKLE]]
    ankle_mid = ankles.mean(axis=0)
    foot_u = ankle_mid[0] if ankles.min() > 0 else (x0 + x1) / 2   # x from the ankles, y from the soles
    gx, gy = to_court(foot_u, y1)
    ax, ay = to_court(*ankle_mid)
    height = height_above_ground((x0 + x1) / 2, y0, gx, gy)
    print(f"  {i:<2} {c:.2f}   ({gx:+6.2f}, {gy:+6.2f})              ({ax:+6.2f}, {ay:+6.2f})   {height:.2f}")
    label = f"#{i} ({gx:+.1f},{gy:+.1f}) {height:.2f}m"
    for colour, thickness in (((0, 0, 0), 4), ((255, 255, 255), 2)):
        cv2.putText(picture, label, (int(x0), int(y1) + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, thickness)
cv2.imwrite("data/people.png", picture)
print("Saved data/people.png")