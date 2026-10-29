import sys
import cv2
import numpy as np
import court
from linemask import line_mask
from findlines import find_lines, homogeneous

# Usage: python 05_fit_court.py VIDEO FRAME FAR_BASELINE NEAR_BASELINE LEFT_SIDELINE RIGHT_SIDELINE
# (the last four are line numbers from script 4; sidelines = the outer, doubles ones)
path, frame_number = sys.argv[1], int(sys.argv[2])
far, near, left, right = (int(a) for a in sys.argv[3:7])

video = cv2.VideoCapture(path)
video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ok, frame = video.read()
mask = line_mask(frame)
lines = find_lines(mask)

# 1. The four outer corners of the court in the image: where the chosen lines cross.
def crossing(i, j):
    p = np.cross(homogeneous(lines[i]["segment"]), homogeneous(lines[j]["segment"]))
    return p[:2] / p[2]

image_corners = np.float32([crossing(far, left), crossing(far, right),
                            crossing(near, left), crossing(near, right)])
L, D = court.HALF_LENGTH, court.HALF_WIDTH_DOUBLES
court_corners = np.float32([[-D, L], [D, L], [-D, -L], [D, -L]])   # same order, in metres

# 2. The homography: the 3x3 matrix that maps court metres to image pixels.
H = cv2.getPerspectiveTransform(court_corners, image_corners)
print("Homography:\n", np.round(H, 3))

# 3. Draw every court line through it.
court_points = court.points_along_lines(0.05).astype(np.float32)
image_points = cv2.perspectiveTransform(court_points.reshape(-1, 1, 2), H).reshape(-1, 2)

# 4. How well does it fit? For each drawn point, the distance to the nearest white mask pixel.
#    distanceTransform gives, for every pixel, the distance to the nearest zero pixel,
#    so we give it an image that is zero exactly on the mask.
distance_to_mask = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
h, w = mask.shape
x, y = image_points[:, 0].astype(int), image_points[:, 1].astype(int)
inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
d = distance_to_mask[y[inside], x[inside]]
print(f"Court points inside the image: {inside.mean() * 100:.0f}%")
print(f"Distance from drawn lines to white pixels: median {np.median(d):.2f} px, "
      f"{(d <= 3).mean() * 100:.0f}% within 3 px")

picture = frame.copy()
for px, py in image_points[inside]:
    cv2.circle(picture, (int(px), int(py)), 1, (0, 0, 255), -1)
for cx, cy in image_corners:
    cv2.circle(picture, (int(cx), int(cy)), 10, (0, 255, 0), 2)
cv2.imwrite("data/court_fit.png", picture)
print("Saved data/court_fit.png")