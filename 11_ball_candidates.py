import sys
import cv2
import numpy as np

if len(sys.argv) < 3:
    sys.exit("Usage: python 11_ball_candidates.py VIDEO FRAME   e.g. python 11_ball_candidates.py data/centre1.mp4 400")
path, frame_number = sys.argv[1], int(sys.argv[2])

YELLOWER_BY = 30               # how much yellower than in both neighbouring frames a pixel must be (0-255)
MIN_AREA, MAX_AREA = 3, 1500   # blob size in pixels: a ball is ~6-15 px across, longer when blurred

# Read the frame before, the frame itself, and the frame after.
video = cv2.VideoCapture(path)
video.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
frames = [video.read()[1] for _ in range(3)]
frame = frames[1]


def yellowness_of(f):
    """(red + green)/2 - blue for every pixel: high for optic yellow, lower for grass, ~0 for white."""
    b, g, r = (f[:, :, c].astype(np.int16) for c in range(3))
    return (r + g) / 2 - b


y_before, yellowness, y_after = (yellowness_of(f) for f in frames)

# Became yellow: much yellower now than in BOTH neighbouring frames.
# The ball arriving over grass raises yellowness; players, white clothes and shadows lower it,
# and grass that was there a moment ago is still there.
yellow = (yellowness - y_before > YELLOWER_BY) & (yellowness - y_after > YELLOWER_BY)

# Join pixels that touch into blobs.
candidates = cv2.morphologyEx(yellow.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
count, labels, stats, centres = cv2.connectedComponentsWithStats(candidates)

print(f"Pixels that became yellow: {yellow.sum()}")
print("Ball candidates:")
print("   x      y    area  width height  yellowness")
picture = frame.copy()
found = 0
for k in range(1, count):                     # label 0 is the background
    x, y, w, h, area = stats[k]
    if not (MIN_AREA <= area <= MAX_AREA):
        continue
    cx, cy = centres[k]
    mean_yellow = yellowness[labels == k].mean()
    print(f"  {cx:6.1f} {cy:6.1f} {area:5d} {w:5d} {h:5d}   {mean_yellow:6.1f}")
    cv2.circle(picture, (int(cx), int(cy)), max(w, h) // 2 + 12, (0, 0, 255), 2)
    found += 1
print(f"{found} candidates")

cv2.imwrite("data/ball_candidates.png", picture)
change_picture = np.zeros_like(frame)
change_picture[yellow] = (0, 0, 255)          # red = became yellow
cv2.imwrite("data/ball_motion.png", change_picture)
print("Saved data/ball_candidates.png and data/ball_motion.png")