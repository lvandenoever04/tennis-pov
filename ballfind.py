import cv2
import numpy as np

YELLOWER_BY = 30               # how much yellower than in both neighbouring frames a pixel must be (0-255)
MIN_AREA, MAX_AREA = 3, 1500   # blob size in pixels


def yellowness_of(f):
    b, g, r = (f[:, :, c].astype(np.int16) for c in range(3))
    return (r + g) / 2 - b


def find_candidates(before, now, after):
    """Blobs that became yellow in `now` compared with both neighbouring frames.
    Returns a list of (x, y, area, yellowness)."""
    y_before, y_now, y_after = yellowness_of(before), yellowness_of(now), yellowness_of(after)
    yellow = (y_now - y_before > YELLOWER_BY) & (y_now - y_after > YELLOWER_BY)
    mask = cv2.morphologyEx(yellow.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, centres = cv2.connectedComponentsWithStats(mask)
    found = []
    for k in range(1, count):
        area = stats[k, 4]
        if MIN_AREA <= area <= MAX_AREA:
            found.append((centres[k][0], centres[k][1], area, y_now[labels == k].mean()))
    return found