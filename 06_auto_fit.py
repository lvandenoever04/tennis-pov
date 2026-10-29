import sys
import time
import itertools
import cv2
import numpy as np
import court
from linemask import line_mask
from findlines import find_lines, homogeneous, angle_of

if len(sys.argv) < 3:
    sys.exit("Usage: python 06_auto_fit.py VIDEO FRAME   e.g. python 06_auto_fit.py data/centre1.mp4 400")
path, frame_number = sys.argv[1], int(sys.argv[2])
video = cv2.VideoCapture(path)
video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ok, frame = video.read()
start_time = time.time()

mask = line_mask(frame)
lines = find_lines(mask)
h, w = mask.shape
distance_to_mask = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
court_points = court.points_along_lines(0.1).astype(np.float32)

L, D, S, V = court.HALF_LENGTH, court.HALF_WIDTH_DOUBLES, court.HALF_WIDTH_SINGLES, court.SERVICE_LINE
ACROSS = {L: "far baseline", V: "far service line", -V: "near service line", -L: "near baseline"}
ALONG = {-D: "left doubles sideline", -S: "left singles sideline", S: "right singles sideline", D: "right doubles sideline"}


def angle_from_horizontal(line):
    a = angle_of(*line["segment"])
    return min(a, 180 - a)


across_lines = [l for l in lines if angle_from_horizontal(l) < 10]
along_lines = [l for l in lines if angle_from_horizontal(l) > 20]
print(f"{len(lines)} lines: {len(across_lines)} across, {len(along_lines)} along")


def crossing(a, b):
    p = np.cross(homogeneous(a["segment"]), homogeneous(b["segment"]))
    return None if abs(p[2]) < 1e-9 else p[:2] / p[2]


def project(H, points):
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2), H).reshape(-1, 2)


def plausible(corners):
    """corners: far-left, far-right, near-left, near-right in the image."""
    fl, fr, nl, nr = corners
    return (fl[1] < nl[1] and fr[1] < nr[1]                     # far end is higher up the image
            and fl[0] < fr[0] and nl[0] < nr[0]                 # left is on the left
            and (nr[0] - nl[0]) > (fr[0] - fl[0]))              # near end looks wider


def court_to_image_score(H):
    """Fraction of court points that land within 3 px of a white pixel (off-screen counts as a miss)."""
    p = project(H, court_points)
    x, y = p[:, 0], p[:, 1]
    inside = (x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1)
    hits = np.zeros(len(p), bool)
    hits[inside] = distance_to_mask[y[inside].astype(int), x[inside].astype(int)] <= 3
    return hits.mean()


def image_to_court_score(H):
    """Fraction of detected line length lying on the drawn court."""
    canvas = np.full((h, w), 255, np.uint8)
    for (x0, y0), (x1, y1) in court.LINES:
        p = project(H, np.float32([[x0, y0], [x1, y1]]))
        cv2.line(canvas, tuple(p[0].astype(int)), tuple(p[1].astype(int)), 0, 3)
    distance_to_court = cv2.distanceTransform(canvas, cv2.DIST_L2, 3)
    explained = total = 0.0
    for line in lines:
        x0, y0, x1, y1 = line["segment"]
        t = np.linspace(0, 1, 20)
        xs = np.clip(x0 + (x1 - x0) * t, 0, w - 1).astype(int)
        ys = np.clip(y0 + (y1 - y0) * t, 0, h - 1).astype(int)
        total += line["length"]
        if np.median(distance_to_court[ys, xs]) <= 4:
            explained += line["length"]
    return explained / total


# 1-3: every guess -> homography -> quick score
guesses = []
for a, b in itertools.combinations(across_lines, 2):
    top, bottom = sorted([a, b], key=lambda l: l["segment"][1] + l["segment"][3])
    for c, d in itertools.combinations(along_lines, 2):
        pc, pd = crossing(bottom, c), crossing(bottom, d)
        if pc is None or pd is None:
            continue
        left, right = (c, d) if pc[0] < pd[0] else (d, c)
        image_corners = [crossing(top, left), crossing(top, right), crossing(bottom, left), crossing(bottom, right)]
        if any(p is None for p in image_corners) or not plausible(image_corners):
            continue
        for y_far, y_near in itertools.combinations(ACROSS, 2):          # pairs with y_far > y_near
            for x_left, x_right in itertools.combinations(ALONG, 2):     # pairs with x_left < x_right
                court_corners = np.float32([[x_left, y_far], [x_right, y_far], [x_left, y_near], [x_right, y_near]])
                H = cv2.getPerspectiveTransform(court_corners, np.float32(image_corners))
                guesses.append((court_to_image_score(H), H, (top, bottom, left, right),
                                (y_far, y_near, x_left, x_right)))
print(f"Tried {len(guesses)} guesses")

# 4: the slower two-way score for the 30 best
guesses.sort(key=lambda g: -g[0])
best = None
for score1, H, chosen, labels in guesses[:30]:
    score2 = image_to_court_score(H)
    total = score1 + score2
    if best is None or total > best[0]:
        best = (total, score1, score2, H, chosen, labels)

total, score1, score2, H, chosen, labels = best
names = [ACROSS[labels[0]], ACROSS[labels[1]], ALONG[labels[2]], ALONG[labels[3]]]
print(f"Best guess (took {time.time() - start_time:.1f} s):")
for line, name in zip(chosen, names):
    number = next(i for i, l in enumerate(lines) if l is line)
    print(f"  line {number} = {name}")
print(f"  court -> image: {score1 * 100:.0f}% of court points on white pixels")
print(f"  image -> court: {score2 * 100:.0f}% of detected line length explained by the court")

picture = frame.copy()
for px, py in project(H, court_points):
    if 0 <= px < w and 0 <= py < h:
        cv2.circle(picture, (int(px), int(py)), 1, (0, 0, 255), -1)
cv2.imwrite("data/auto_fit.png", picture)
print("Saved data/auto_fit.png")