import sys
import cv2
import numpy as np
from linemask import line_mask

path = sys.argv[1]
frame_number = int(sys.argv[2])
video = cv2.VideoCapture(path)
video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ok, frame = video.read()

from findlines import find_lines, angle_of

mask = line_mask(frame)
lines = find_lines(mask)
print(f"Found {len(lines)} lines:")
for n, line in enumerate(lines):
    print(f"  line {n}: angle {angle_of(*line['segment']):5.1f} deg, total length {line['length']:.0f} px")

# 3. Draw each line extended across the whole image, numbered, in its own colour.
picture = frame.copy()
rng = np.random.default_rng(0)
for n, line in enumerate(lines):
    x0, y0, x1, y1 = line["segment"]
    d = np.array([x1 - x0, y1 - y0]) / np.hypot(x1 - x0, y1 - y0)
    m = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    p, q = (m - 3000 * d).astype(int), (m + 3000 * d).astype(int)
    colour = tuple(int(c) for c in rng.integers(80, 256, 3))
    cv2.line(picture, tuple(p), tuple(q), colour, 2)
    cv2.putText(picture, str(n), tuple(m.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3)
cv2.imwrite("data/lines.png", picture)
print("Saved data/lines.png")