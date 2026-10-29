import sys
import cv2
import numpy as np
from linemask import line_mask

path = sys.argv[1]
frame_number = int(sys.argv[2])
video = cv2.VideoCapture(path)
video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ok, frame = video.read()

MIN_SEGMENT_LENGTH = 150   # px: ignore short bits (text, clothing)
MAX_GAP = 20               # px: join pieces of a line broken by small gaps
SAME_ANGLE = 2.0           # degrees: segments within this angle can be the same line
SAME_LINE = 6.0            # px: ...and if they lie within this distance of each other

mask = line_mask(frame).astype(np.uint8) * 255

# 1. Hough transform: find straight segments made of mask pixels.
found = cv2.HoughLinesP(mask, rho=1, theta=np.pi / 360, threshold=80,
                        minLineLength=MIN_SEGMENT_LENGTH, maxLineGap=MAX_GAP)
segments = [] if found is None else found.reshape(-1, 4).astype(float)
print(f"Hough found {len(segments)} segments")

# 2. Merge segments that belong to the same line, longest first.
def angle_of(x0, y0, x1, y1):
    return np.degrees(np.arctan2(y1 - y0, x1 - x0)) % 180

def distance_to_line(px, py, x0, y0, x1, y1):
    """Perpendicular distance from point (px, py) to the infinite line through (x0,y0)-(x1,y1)."""
    dx, dy = x1 - x0, y1 - y0
    return abs(dy * (px - x0) - dx * (py - y0)) / np.hypot(dx, dy)

segments = sorted(segments, key=lambda s: -np.hypot(s[2] - s[0], s[3] - s[1]))
lines = []   # each: {"segment": the longest piece, "length": total length of all pieces}
for s in segments:
    x0, y0, x1, y1 = s
    length = np.hypot(x1 - x0, y1 - y0)
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    for line in lines:
        angle_diff = abs(angle_of(*s) - angle_of(*line["segment"]))
        angle_diff = min(angle_diff, 180 - angle_diff)
        if angle_diff < SAME_ANGLE and distance_to_line(mx, my, *line["segment"]) < SAME_LINE:
            line["length"] += length
            break
    else:   # no existing line matched: this segment starts a new line
        lines.append({"segment": s, "length": length})

print(f"Merged into {len(lines)} lines:")
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