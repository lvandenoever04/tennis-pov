import sys
import time
import cv2
import numpy as np
import court
from autofit import auto_fit, project

if len(sys.argv) < 3:
    sys.exit("Usage: python 06_auto_fit.py VIDEO FRAME   e.g. python 06_auto_fit.py data/centre1.mp4 400")
path, frame_number = sys.argv[1], int(sys.argv[2])
video = cv2.VideoCapture(path)
video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ok, frame = video.read()

start_time = time.time()
fit = auto_fit(frame)
if fit is None:
    sys.exit("No court found")
print(f"Best guess (took {time.time() - start_time:.1f} s):")
for line, name in zip(fit["chosen"], fit["names"]):
    number = next(i for i, l in enumerate(fit["lines"]) if l is line)
    print(f"  line {number} = {name}")
print(f"  court -> image: {fit['score1'] * 100:.0f}% of court points on white pixels")
print(f"  image -> court: {fit['score2'] * 100:.0f}% of detected line length explained by the court")

h, w = frame.shape[:2]
picture = frame.copy()
for px, py in project(fit["H"], court.points_along_lines(0.05)):
    if 0 <= px < w and 0 <= py < h:
        cv2.circle(picture, (int(px), int(py)), 1, (0, 0, 255), -1)
cv2.imwrite("data/auto_fit.png", picture)
print("Saved data/auto_fit.png")