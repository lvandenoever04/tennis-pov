import sys
import cv2
import numpy as np
from scipy.optimize import least_squares
import court

if len(sys.argv) < 2:
    sys.exit("Usage: python 08_camera_3d.py VIDEO   e.g. python 08_camera_3d.py data/centre1.mp4")
path = sys.argv[1]

NET_POST_X = court.HALF_WIDTH_SINGLES + 0.914   # posts stand 0.914 m outside the doubles sidelines
NET_POST_HEIGHT = 1.07                          # m
NET_CENTRE_HEIGHT = 0.914                       # m (held down by the centre strap)

track = np.load("data/court_track.npz")
H = track["homographies"][0]
video = cv2.VideoCapture(path)
video.set(cv2.CAP_PROP_POS_FRAMES, int(track["first"]))
ok, frame = video.read()
h, w = frame.shape[:2]
cx, cy = w / 2, h / 2      # assume the camera's centre of view is the middle of the image


def camera_matrix(f):
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])


def project(f, R, C, points3d):
    """3D points (metres) -> pixels. R turns world directions into camera directions; C is the camera position."""
    x = (camera_matrix(f) @ (R @ (np.asarray(points3d, float) - C).T)).T
    return x[:, :2] / x[:, 2:3]


# ---- 1. Direct estimate from the homography ----
# Shift pixel coordinates so (0, 0) is the image centre. Then the homography's columns are:
#   column 1 = f * (direction of the court's x axis, seen by the camera)
#   column 2 = f * (direction of the court's y axis)
#   column 3 = f * (where the court's centre is, relative to the camera)
# (each up to one common scale factor).
M = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]]) @ H
h1, h2, h3 = M[:, 0], M[:, 1], M[:, 2]

# The two court axes are at right angles and the same length in reality. Each fact gives one equation
# in the unknown 1/f^2; we solve both together (least squares).
a = np.array([h1[0] * h2[0] + h1[1] * h2[1], h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2])
b = np.array([h1[2] * h2[2], h1[2] ** 2 - h2[2] ** 2])
f = 1 / np.sqrt(-(a @ b) / (a @ a))

# Undo the zoom, then rescale so the axis directions have length 1.
r1, r2, t = h1.copy(), h2.copy(), h3.copy()
r1[:2] /= f; r2[:2] /= f; t[:2] /= f
scale = 2 / (np.linalg.norm(r1) + np.linalg.norm(r2))
r1, r2, t = r1 * scale, r2 * scale, t * scale
if t[2] < 0:                 # the court must be in front of the camera, not behind it
    r1, r2, t = -r1, -r2, -t
R = np.stack([r1, r2, np.cross(r1, r2)], axis=1)   # third axis (up) is at right angles to both
U, _, Vt = np.linalg.svd(R)
R = U @ Vt                   # nearest exact rotation (removes small inconsistencies)
C = -R.T @ t                 # camera position in court coordinates

# ---- 2. Refine: the 7 camera numbers that draw the court where the homography does ----
court_points = court.points_along_lines(0.25)
target = np.c_[court_points, np.ones(len(court_points))] @ H.T
target = target[:, :2] / target[:, 2:3]
ground = np.c_[court_points, np.zeros(len(court_points))]


def unpack(p):
    R, _ = cv2.Rodrigues(p[:3])      # 3 numbers -> a rotation (axis direction x angle)
    return p[3], R, p[4:7]


def mismatch(p):
    f, R, C = unpack(p)
    return (project(f, R, C, ground) - target).ravel()


before = np.hypot(*mismatch(np.r_[cv2.Rodrigues(R)[0].ravel(), f, C]).reshape(-1, 2).T).max()
result = least_squares(mismatch, np.r_[cv2.Rodrigues(R)[0].ravel(), f, C],
                       x_scale=np.r_[0.01, 0.01, 0.01, 50, 0.5, 0.5, 0.5])
f, R, C = unpack(result.x)
e = np.hypot(*mismatch(result.x).reshape(-1, 2).T)
print(f"Camera vs homography: max {before:.2f} px before refining, median {np.median(e):.2f} / max {e.max():.2f} px after")

# ---- Report ----
view = R[2]                  # the direction the camera looks, in court coordinates
print(f"Camera position: {C[0]:+.2f} m across, {C[1]:+.2f} m along (negative = behind the near baseline), "
      f"{C[2]:.2f} m up")
print(f"Looking {np.degrees(-np.arcsin(view[2])):.1f} deg down, "
      f"{np.degrees(np.arctan2(view[0], view[1])):+.2f} deg left/right of straight down the court")
print(f"Zoom: focal length {f:.0f} px = {np.degrees(2 * np.arctan(w / 2 / f)):.1f} deg wide field of view")
far = court.HALF_LENGTH
top, foot = project(f, R, C, [[0, far, 1.85], [0, far, 0]])
print(f"A 1.85 m player on the far baseline is {foot[1] - top[1]:.0f} px tall")

# ---- 3. Independent check: the net ----
xs = np.linspace(-NET_POST_X, NET_POST_X, 300)
tape = np.c_[xs, np.zeros_like(xs), NET_CENTRE_HEIGHT + (NET_POST_HEIGHT - NET_CENTRE_HEIGHT) * (xs / NET_POST_X) ** 2]
tape_px = project(f, R, C, tape)
mask = cv2.imread("data/static_mask.png", cv2.IMREAD_GRAYSCALE) > 0
distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
inside = (tape_px[:, 0] >= 0) & (tape_px[:, 0] < w) & (tape_px[:, 1] >= 0) & (tape_px[:, 1] < h)
d = distance[tape_px[inside, 1].astype(int), tape_px[inside, 0].astype(int)]
print(f"Net tape check: drawn net is a median {np.median(d):.1f} px from white pixels "
      f"({(d <= 3).mean() * 100:.0f}% within 3 px)")

np.savez("data/camera.npz", f=f, R=R, C=C, cx=cx, cy=cy, width=w, height=h)
print("Saved data/camera.npz")

picture = frame.copy()
for (x0, y0), (x1, y1) in court.LINES:
    p = project(f, R, C, [[x0, y0, 0], [x1, y1, 0]])
    cv2.line(picture, tuple(np.round(p[0]).astype(int)), tuple(np.round(p[1]).astype(int)), (0, 0, 255), 1, cv2.LINE_AA)
cv2.polylines(picture, [np.round(tape_px).astype(np.int32)], False, (255, 255, 0), 1, cv2.LINE_AA)
for sx in (-NET_POST_X, NET_POST_X):
    p = project(f, R, C, [[sx, 0, 0], [sx, 0, NET_POST_HEIGHT]])
    cv2.line(picture, tuple(np.round(p[0]).astype(int)), tuple(np.round(p[1]).astype(int)), (0, 255, 255), 2, cv2.LINE_AA)
cv2.imwrite("data/camera_check.png", picture)
print("Saved data/camera_check.png")