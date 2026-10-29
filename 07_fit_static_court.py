import sys
import time
import cv2
import numpy as np
from scipy.optimize import least_squares
import court
from linemask import line_mask
from autofit import auto_fit_mask

if len(sys.argv) < 4:
    sys.exit("Usage: python 07_fit_static_court.py VIDEO FIRST_FRAME LAST_FRAME   e.g. ... data/centre1.mp4 0 845")
path, first, last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

SAMPLE_EVERY = 10          # frames combined into the clean line mask: every 10th
KEEP_FRACTION = 0.5        # a pixel counts as painted line if it is white in at least half of them
SEARCH_RADII = [10, 7, 7]  # px: how near a drawn line a white pixel must be to belong to it.
                           # Must stay wider than half the thickest painted line (~4 px), or one
                           # side of the paint gets cut off and the fit is pulled off-centre.
CHECK_EVERY = 5            # frames checked for camera movement: every 5th
MOVED = 1.0                # px: a re-fit further than this from the static fit means the camera moved

L, D = court.HALF_LENGTH, court.HALF_WIDTH_DOUBLES
COURT_CORNERS = np.array([[-D, L], [D, L], [-D, -L], [D, -L]])   # far-left, far-right, near-left, near-right


# OpenCV's getPerspectiveTransform / perspectiveTransform use 32-bit numbers (~7 significant digits).
# The optimiser probes with tiny nudges (~0.00000001 px) that 32-bit rounding erases, so it would
# see no change and never move. These two do the same maths in 64-bit.

def homography_from_corners(corners):
    """The homography mapping the court's 4 outer corners (metres) to `corners` (pixels)."""
    A, b = [], []
    for (X, Y), (u, v) in zip(COURT_CORNERS, np.asarray(corners, float).reshape(4, 2)):
        A.append([X, Y, 1, 0, 0, 0, -u * X, -u * Y])
        b.append(u)
        A.append([0, 0, 0, X, Y, 1, -v * X, -v * Y])
        b.append(v)
    h = np.linalg.solve(np.array(A), np.array(b))
    return np.append(h, 1.0).reshape(3, 3)


def project64(H, points):
    """Court points (metres) -> image points (pixels)."""
    p = np.asarray(points, float).reshape(-1, 2)
    q = np.c_[p, np.ones(len(p))] @ H.T
    return q[:, :2] / q[:, 2:3]


def read_frame(video, i):
    video.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, frame = video.read()
    return frame


def pixels_near_lines(H, xs, ys, radius):
    """For each court line: the white pixels within `radius` px of where it is drawn (and alongside it)."""
    groups = []
    for (x0, y0), (x1, y1) in court.LINES:
        (ax, ay), (bx, by) = project64(H, [[x0, y0], [x1, y1]])
        dx, dy = bx - ax, by - ay
        along = ((xs - ax) * dx + (ys - ay) * dy) / (dx * dx + dy * dy)
        across = np.abs(dy * (xs - ax) - dx * (ys - ay)) / np.hypot(dx, dy)
        keep = (along >= 0) & (along <= 1) & (across <= radius)
        groups.append(np.stack([xs[keep], ys[keep]], axis=1))
    return groups


def residuals(corner_shift, corners, groups):
    """Signed distance (px) of every collected pixel from its court line: + on one side, - on the other."""
    H = homography_from_corners(corners + corner_shift.reshape(4, 2))
    out = []
    for ((x0, y0), (x1, y1)), pts in zip(court.LINES, groups):
        if len(pts) == 0:
            continue
        (ax, ay), (bx, by) = project64(H, [[x0, y0], [x1, y1]])
        dx, dy = bx - ax, by - ay
        out.append((dy * (pts[:, 0] - ax) - dx * (pts[:, 1] - ay)) / np.hypot(dx, dy))
    return np.concatenate(out)


def fit_to_line_centres(corners, mask, radii=SEARCH_RADII):
    """Move the court until every drawn line runs through the middle of its painted line."""
    ys, xs = np.nonzero(mask)
    xs, ys = xs.astype(np.float64), ys.astype(np.float64)
    for radius in radii:
        groups = pixels_near_lines(homography_from_corners(corners), xs, ys, radius)
        result = least_squares(residuals, np.zeros(8), args=(corners, groups), loss="soft_l1", f_scale=1.0)
        corners = corners + result.x.reshape(4, 2)
    return corners


def describe(corners):
    return " ".join(f"({x:7.1f},{y:6.1f})" for x, y in corners)


video = cv2.VideoCapture(path)
start_time = time.time()

# 1. One clean line mask from the whole rally: players and ball move, painted lines don't.
samples = list(range(first, last + 1, SAMPLE_EVERY))
count = None
for i in samples:
    m = line_mask(read_frame(video, i)).astype(np.uint16)
    count = m if count is None else count + m
static_mask = count >= KEEP_FRACTION * len(samples)
cv2.imwrite("data/static_mask.png", static_mask.astype(np.uint8) * 255)
print(f"Combined {len(samples)} frames into data/static_mask.png")

# 2. Rough fit: script 6's search, on the clean mask.
rough = project64(auto_fit_mask(static_mask)["H"], COURT_CORNERS)
print(f"Rough fit corners:   {describe(rough)}")

# 3. Precise fit: through the centres of the painted lines.
corners = fit_to_line_centres(rough, static_mask)
H = homography_from_corners(corners)
print(f"Precise fit corners: {describe(corners)}")
print(f"  (moved up to {np.abs(corners - rough).max():.1f} px from the rough fit)")

# 4. Did the camera move? Re-fit individual frames starting from the static fit and see how far they go.
checked, movement = [], []
for i in range(first, last + 1, CHECK_EVERY):
    c = fit_to_line_centres(corners, line_mask(read_frame(video, i)), radii=[7])
    checked.append(i)
    movement.append(np.abs(c - corners).max())
movement = np.array(movement)
worst = int(np.argmax(movement))
print(f"Camera movement check on {len(checked)} frames: typical {np.median(movement):.2f} px, "
      f"worst {movement[worst]:.2f} px (frame {checked[worst]})")
moved = [f for f, m in zip(checked, movement) if m > MOVED]
if moved:
    print(f"  WARNING: {len(moved)} checked frames are more than {MOVED} px off: {moved[:10]}")
else:
    print("  The camera is static for this rally: one homography for every frame")
print(f"Took {time.time() - start_time:.0f} s")

# 5. Save: the same homography for every frame (later scripts expect one per frame).
n = last - first + 1
np.savez("data/court_track.npz", homographies=np.repeat(H[None], n, axis=0), first=first, last=last,
         checked_frames=np.array(checked), movement=movement)
print("Saved data/court_track.npz")

picture = read_frame(video, first)
for (x0, y0), (x1, y1) in court.LINES:
    p = project64(H, [[x0, y0], [x1, y1]])
    cv2.line(picture, tuple(np.round(p[0] * 16).astype(int)), tuple(np.round(p[1] * 16).astype(int)),
             (0, 0, 255), 1, cv2.LINE_AA, shift=4)
cv2.imwrite("data/static_fit.png", picture)
print("Saved data/static_fit.png")