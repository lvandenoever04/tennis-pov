"""Find the court in one frame with no prior knowledge: try every guess, keep the best."""
import itertools
import cv2
import numpy as np
import court
from linemask import line_mask
from findlines import find_lines, homogeneous, angle_of

L, D, S, V = court.HALF_LENGTH, court.HALF_WIDTH_DOUBLES, court.HALF_WIDTH_SINGLES, court.SERVICE_LINE
ACROSS = {L: "far baseline", V: "far service line", -V: "near service line", -L: "near baseline"}
ALONG = {-D: "left doubles sideline", -S: "left singles sideline", S: "right singles sideline", D: "right doubles sideline"}
COURT_POINTS = court.points_along_lines(0.1).astype(np.float32)


def project(H, points):
    """Court points (metres) -> image points (pixels)."""
    return cv2.perspectiveTransform(np.float32(points).reshape(-1, 1, 2), H).reshape(-1, 2)


def angle_from_horizontal(line):
    a = angle_of(*line["segment"])
    return min(a, 180 - a)


def crossing(a, b):
    p = np.cross(homogeneous(a["segment"]), homogeneous(b["segment"]))
    return None if abs(p[2]) < 1e-9 else p[:2] / p[2]


def plausible(corners):
    fl, fr, nl, nr = corners
    return (fl[1] < nl[1] and fr[1] < nr[1] and fl[0] < fr[0] and nl[0] < nr[0]
            and (nr[0] - nl[0]) > (fr[0] - fl[0]))


def court_to_image_score(H, distance_to_mask):
    h, w = distance_to_mask.shape
    p = project(H, COURT_POINTS)
    x, y = p[:, 0], p[:, 1]
    inside = (x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1)
    hits = np.zeros(len(p), bool)
    hits[inside] = distance_to_mask[y[inside].astype(int), x[inside].astype(int)] <= 3
    return hits.mean()


def image_to_court_score(H, lines, shape):
    h, w = shape
    canvas = np.full((h, w), 255, np.uint8)
    for (x0, y0), (x1, y1) in court.LINES:
        p = project(H, [[x0, y0], [x1, y1]])
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


def auto_fit(frame, keep=30):
    """Returns a dict: H (homography), score1, score2, lines (all detected), chosen, names. None if nothing fits."""
    return auto_fit_mask(line_mask(frame), keep)


def auto_fit_mask(mask, keep=30):
    """The same search, on a line mask you already have."""
    lines = find_lines(mask)
    distance_to_mask = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
    across_lines = [l for l in lines if angle_from_horizontal(l) < 10]
    along_lines = [l for l in lines if angle_from_horizontal(l) > 20]

    guesses = []
    for a, b in itertools.combinations(across_lines, 2):
        top, bottom = sorted([a, b], key=lambda l: l["segment"][1] + l["segment"][3])
        for c, d in itertools.combinations(along_lines, 2):
            pc, pd = crossing(bottom, c), crossing(bottom, d)
            if pc is None or pd is None:
                continue
            left, right = (c, d) if pc[0] < pd[0] else (d, c)
            corners = [crossing(top, left), crossing(top, right), crossing(bottom, left), crossing(bottom, right)]
            if any(p is None for p in corners) or not plausible(corners):
                continue
            for y_far, y_near in itertools.combinations(ACROSS, 2):
                for x_left, x_right in itertools.combinations(ALONG, 2):
                    court_corners = np.float32([[x_left, y_far], [x_right, y_far], [x_left, y_near], [x_right, y_near]])
                    H = cv2.getPerspectiveTransform(court_corners, np.float32(corners))
                    guesses.append((court_to_image_score(H, distance_to_mask), H, (top, bottom, left, right),
                                    (y_far, y_near, x_left, x_right)))
    if not guesses:
        return None
    guesses.sort(key=lambda g: -g[0])
    best = None
    for score1, H, chosen, labels in guesses[:keep]:
        score2 = image_to_court_score(H, lines, mask.shape)
        if best is None or score1 + score2 > best["score1"] + best["score2"]:
            names = [ACROSS[labels[0]], ACROSS[labels[1]], ALONG[labels[2]], ALONG[labels[3]]]
            best = {"H": H, "score1": score1, "score2": score2, "lines": lines, "chosen": chosen, "names": names}
    return best