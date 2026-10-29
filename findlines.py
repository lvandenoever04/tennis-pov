import cv2
import numpy as np

MIN_SEGMENT_LENGTH = 150
MAX_GAP = 20
SAME_ANGLE = 2.0
SAME_LINE = 6.0


def angle_of(x0, y0, x1, y1):
    return np.degrees(np.arctan2(y1 - y0, x1 - x0)) % 180


def distance_to_line(px, py, x0, y0, x1, y1):
    dx, dy = x1 - x0, y1 - y0
    return abs(dy * (px - x0) - dx * (py - y0)) / np.hypot(dx, dy)


def find_lines(mask):
    """Straight lines in a line mask, longest first: list of {"segment": (x0,y0,x1,y1), "length": px}."""
    found = cv2.HoughLinesP(mask.astype(np.uint8) * 255, rho=1, theta=np.pi / 360, threshold=80,
                            minLineLength=MIN_SEGMENT_LENGTH, maxLineGap=MAX_GAP)
    segments = [] if found is None else found.reshape(-1, 4).astype(float)
    segments = sorted(segments, key=lambda s: -np.hypot(s[2] - s[0], s[3] - s[1]))
    lines = []
    for s in segments:
        length = np.hypot(s[2] - s[0], s[3] - s[1])
        mx, my = (s[0] + s[2]) / 2, (s[1] + s[3]) / 2
        for line in lines:
            diff = abs(angle_of(*s) - angle_of(*line["segment"]))
            if min(diff, 180 - diff) < SAME_ANGLE and distance_to_line(mx, my, *line["segment"]) < SAME_LINE:
                line["length"] += length
                break
        else:
            lines.append({"segment": s, "length": length})
    return lines


def homogeneous(segment):
    """The infinite line through a segment, as (a, b, c) with a*x + b*y + c = 0.
    Crossing two of these (np.cross) gives the point where the lines intersect."""
    x0, y0, x1, y1 = segment
    return np.cross([x0, y0, 1.0], [x1, y1, 1.0])