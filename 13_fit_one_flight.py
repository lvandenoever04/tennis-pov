import sys
import cv2
import numpy as np
from scipy.optimize import least_squares
import court
from geometry import Camera

if len(sys.argv) < 4:
    sys.exit("Usage: python 13_fit_one_flight.py VIDEO FIRST_FRAME LAST_FRAME [BOUNCE_FRAME]\n"
             "   FIRST_FRAME = first frame after the hit, LAST_FRAME = last frame before the bounce\n"
             "   BOUNCE_FRAME = when it bounces (default LAST_FRAME + 0.5), or 'none' if the bounce can't be seen")
path, flight_first, flight_last = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
bounce_arg = sys.argv[4] if len(sys.argv) > 4 else str(flight_last + 0.5)
seen_bounce = None if bounce_arg == "none" else float(bounce_arg)

GRAVITY = 9.81             # m/s^2
DRAG = 0.0204              # 1/m: air drag on a tennis ball, = 0.5 * air density * drag coefficient * area / mass
SPIN_RANGE = (-12.0, 4.0)  # m/s^2: spin's push at the start (negative = topspin dips, positive = backspin floats)
BALL_RADIUS = 0.033        # m
PIXEL_NOISE = 2.0          # px: typical error of a ball detection
BLUR_ERROR = 0.1           # extra error ALONG the direction of motion, as a fraction of the ball's image
                           # movement per frame: a fast ball is a streak, and TrackNet's point wanders along it
OUTLIER = 8.0              # px: a detection this far from the fitted path is a mistake (TrackNet on the wrong thing)
HEIGHT_RANGE = (0.2, 2.8)  # m: lowest and highest contact heights
WALL = 100.0               # px per metre: penalty for the impossible (contact too low/high, ball underground)
CONTACT_AHEAD = 0.5        # m: the racket meets the ball about this far in front of the player's body...
CONTACT_SPREAD = 0.75      # m: ...give or take this much (along the court, i.e. towards/away from the camera)
BOUNCE_SPREAD = 0.5        # frames: how sure we are of the bounce time (it happens somewhere between two frames)
EXPOSURE = 0.5             # frames: the shutter is open about half a frame, so a fast ball is a streak, and
                           # TrackNet marks its leading end: where the ball is at the END of the exposure

# Why the contact prior is needed: one camera can't see depth directly. Take any flight, move it closer to
# the camera and shrink it by the same factor: it lands on exactly the same pixels, only with weaker
# accelerations. Gravity is fixed, but the spin isn't, so "nearer + backspin" and "further + topspin" fit the
# detections almost equally well (a 10 m slide changes the error by ~1 px^2). Two things decide it:
#  - where the player was standing when they hit it (the contact prior), and
#  - when it bounces: sliding the flight along the camera rays also moves it up or down (the camera looks
#    down), so only one depth has the ball reach the ground at the moment we see it bounce.

cam = Camera()
b = np.load("data/ball2d.npz")
clip_first, fps = int(b["range"][0]), float(b["fps"])
frames = np.arange(flight_first, flight_last + 1) - clip_first
frames = frames[b["source"][frames] == 1]              # only real detections, not filled-in ones
uv_all = b["position"][frames]

# Time 0 is the moment of contact. The ball was still on the incoming side in the frame before FIRST_FRAME,
# and the streak in FIRST_FRAME starts at the racket, so contact is about when FIRST_FRAME's exposure opened.
# Each detection is the end of its frame's exposure, i.e. EXPOSURE frames after that.
t_all = (frames + clip_first - flight_first + EXPOSURE) / fps
print(f"{len(frames)} detections over {t_all[-1] - t_all[0]:.2f} s")


def acceleration(v, spin, speed0):
    """Gravity down, drag opposite to the motion (growing with speed squared), and spin's Magnus push:
    at right angles to the motion, in the vertical plane, also growing with speed squared.
    `spin` is the push at the starting speed: negative = topspin (pushes down), positive = backspin (lifts)."""
    up = np.array([0.0, 0.0, 1.0])
    across = up - v * (v @ up) / (v @ v)              # the 'up' direction, made perpendicular to the motion
    across /= np.linalg.norm(across)
    magnus = spin * (v @ v) / speed0 ** 2 * across
    return np.array([0, 0, -GRAVITY]) - DRAG * np.linalg.norm(v) * v + magnus


def simulate(p0, v0, spin, times, step=1 / 240, with_velocity=False):
    """Ball positions (and velocities, if asked) at the given times (seconds after contact, increasing),
    starting at p0 with velocity v0."""
    out = np.empty((len(times), 3))
    out_v = np.empty((len(times), 3))
    p, v, now = np.array(p0, float), np.array(v0, float), 0.0
    speed0 = np.linalg.norm(v0)
    for i, target in enumerate(times):
        while now < target - 1e-9:
            h = min(step, target - now)
            v_mid = v + acceleration(v, spin, speed0) * h / 2  # half a step ahead, for accuracy
            p = p + v_mid * h
            v = v + acceleration(v_mid, spin, speed0) * h
            now += h
        out[i], out_v[i] = p, v
    return (out, out_v) if with_velocity else out


def point_at_height(u, v, z):
    """Where the camera ray through pixel (u, v) is z metres above the ground."""
    ray = cam.R.T @ np.array([(u - cam.cx) / cam.f, (v - cam.cy) / cam.f, 1.0])
    return cam.C + (z - cam.C[2]) / ray[2] * ray


# Who hit it? The player on the side where the flight starts. They hit it a little in front of their body.
players = np.load("data/players.npz")
side = "near" if point_at_height(*uv_all[0], 1.0)[1] < 0 else "far"
forward = 1.0 if side == "near" else -1.0               # the near player hits towards +y, the far one towards -y
hitter = players[f"{side}_position"][flight_first - clip_first]
contact_y = hitter[1] + forward * CONTACT_AHEAD
print(f"Hit by the {side} player, standing at ({hitter[0]:+.2f}, {hitter[1]:+.2f}) m")
if seen_bounce is not None:
    t_bounce = (seen_bounce - flight_first + EXPOSURE) / fps      # same clock as the detections
    print(f"Bounce seen at frame {seen_bounce:.1f}")
else:
    print("Bounce not seen: depth comes from the contact prior only")


def errors(params, t, uv):
    """Pixel distance between the simulated flight and the detections, plus the contact prior and the bounce.
    params = contact position (3), velocity at contact (3), spin (1).
    The extra terms are written in pixels too: being one 'spread' off costs as much as one PIXEL_NOISE miss."""
    p0 = params[:3]
    prior = (p0[1] - contact_y) / CONTACT_SPREAD * PIXEL_NOISE
    outside_height = max(0.0, HEIGHT_RANGE[0] - p0[2]) + max(0.0, p0[2] - HEIGHT_RANGE[1])
    if seen_bounce is None:
        X, V = simulate(p0, params[3:6], params[6], t, with_velocity=True)
        bounce = 0.0
    else:
        # Height above the ground at the bounce moment, in "frames early or late" (height / falling speed)
        X, V = simulate(p0, params[3:6], params[6], np.r_[t, t_bounce], with_velocity=True)
        bounce = (X[-1, 2] - BALL_RADIUS) / max(-V[-1, 2] / fps, 0.01) / BOUNCE_SPREAD * PIXEL_NOISE
        X, V = X[:-1], V[:-1]
    # Split each miss into along and across the ball's motion in the picture, and forgive more along it.
    miss = cam.project(X) - uv
    motion = cam.project(X + V / fps) - cam.project(X)            # image movement in one frame (px)
    speed = np.linalg.norm(motion, axis=1)
    along = (miss * motion).sum(1) / speed
    across = (miss[:, 0] * motion[:, 1] - miss[:, 1] * motion[:, 0]) / speed
    along_noise = np.hypot(PIXEL_NOISE, BLUR_ERROR * speed)
    pixels = np.c_[along * PIXEL_NOISE / along_noise, across].ravel()
    # Before the bounce the ball is in the air: a path that dips under the grass can still land on the right
    # pixels (it's just further along the same camera rays), so rule that out explicitly.
    underground = np.minimum(X[:, 2] - BALL_RADIUS, 0.0)
    return np.r_[pixels, prior, bounce, WALL * outside_height, WALL * underground]


# Starting guess: contact on the first detection's camera ray, level with the player; the ball reaches the
# last detection's ray 0.5 m up in a straight line, plus the upward speed needed to fight gravity; no spin.
start = point_at_height(*uv_all[0], 1.0)
start = cam.C + (contact_y - cam.C[1]) / (start[1] - cam.C[1]) * (start - cam.C)
end = point_at_height(*uv_all[-1], 0.5)
v_guess = (end - start) / t_all[-1]
v_guess[2] += GRAVITY * t_all[-1] / 2
lower = [-15, -25, 0.0, -80, -80, -80, SPIN_RANGE[0]]
upper = [15, 25, 15.0, 80, 80, 80, SPIN_RANGE[1]]
guess = np.r_[start, v_guess, -1.0]

# Fit, then throw out the worst detection if it is clearly not on the path, and fit again, until all fit.
# soft_l1 while there may still be mistakes (so they can't bend the curve much); plain least squares at the end.
keep = np.ones(len(t_all), bool)
while True:
    fit = least_squares(errors, guess, args=(t_all[keep], uv_all[keep]), bounds=(lower, upper),
                        loss="soft_l1", f_scale=PIXEL_NOISE, x_scale="jac")
    guess = fit.x
    all_errors = np.hypot(*(cam.project(simulate(fit.x[:3], fit.x[3:6], fit.x[6], t_all)) - uv_all).T)
    worst = np.argmax(np.where(keep, all_errors, -1))
    if all_errors[worst] <= OUTLIER or keep.sum() <= 8:
        break
    keep[worst] = False
    print(f"  rejected frame {frames[worst] + clip_first}: {all_errors[worst]:.0f} px from the path")
t, uv = t_all[keep], uv_all[keep]
fit = least_squares(errors, guess, args=(t, uv), bounds=(lower, upper), x_scale="jac")
p0, v0, spin = fit.x[:3], fit.x[3:6], fit.x[6]
pixel_error = np.hypot(*(cam.project(simulate(p0, v0, spin, t)) - uv).T)

# How certain is it? If detections are ~2 px off (and the physics is right), how much could the start vary?
J = fit.jac
spread = np.sqrt(np.diag(PIXEL_NOISE ** 2 * np.linalg.inv(J.T @ J)))

print(f"Fit: {len(t)} detections are a median {np.median(pixel_error):.1f} px from the 3D path "
      f"(worst {pixel_error.max():.1f} px)")
if np.median(pixel_error) > 3:
    print("  WARNING: poor fit - the range probably includes a hit or a bounce; try a shorter range")
print(f"Contact: ({p0[0]:+.2f}, {p0[1]:+.2f}, {p0[2]:.2f}) m   +/- ({spread[0]:.2f}, {spread[1]:.2f}, {spread[2]:.2f}) m")
print(f"  {np.hypot(*(p0[:2] - hitter)):.2f} m from the hitter's body, "
      f"{forward * (p0[1] - hitter[1]):+.2f} m in front of it (prior: {CONTACT_AHEAD:+.2f} +/- {CONTACT_SPREAD:.2f})")
print(f"Speed at contact: {np.linalg.norm(v0):.1f} m/s = {np.linalg.norm(v0) * 3.6:.0f} km/h")
kind = "topspin (dips)" if spin < -0.5 else ("backspin (floats)" if spin > 0.5 else "little spin")
print(f"Spin: {spin:+.1f} m/s^2 ({spin / GRAVITY:+.2f} g) at contact - {kind}   +/- {spread[6]:.1f}")

# Follow the flight a little beyond the last detection: net crossing, highest point, bounce.
dense_t = np.linspace(0, t[-1] + 1.0, 600)
path3d = simulate(p0, v0, spin, dense_t)
print(f"Highest point: {path3d[:, 2].max():.2f} m")
cross = np.nonzero(np.diff(np.sign(path3d[:, 1])))[0]
if len(cross):
    print(f"Crosses the net line at {path3d[cross[0], 2]:.2f} m high (net is 0.914 m in the middle, 1.07 m at the posts)")
landing = np.nonzero(path3d[:, 2] <= BALL_RADIUS)[0]
bounce = None
if len(landing):
    bounce = path3d[landing[0]]
    bounce_frame = flight_first + dense_t[landing[0]] * fps - EXPOSURE
    inside = abs(bounce[0]) <= court.HALF_WIDTH_SINGLES and abs(bounce[1]) <= court.HALF_LENGTH
    seen = "" if seen_bounce is None else f" (seen at {seen_bounce:.1f})"
    print(f"Bounces at ({bounce[0]:+.2f}, {bounce[1]:+.2f}) m, around frame {bounce_frame:.1f}{seen} - "
          f"{'inside' if inside else 'outside'} the singles court")
    path3d = path3d[:landing[0] + 1]

# Picture 1: the middle frame, with the detections used (yellow), rejected ones (magenta), the next few
# detections after the range (cyan: should start near the bounce), the 3D path seen by the camera (red),
# its shadow straight below on the ground (white), the bounce (green) and the hitter's position (blue).
video = cv2.VideoCapture(path)
video.set(cv2.CAP_PROP_POS_FRAMES, (flight_first + flight_last) // 2)
ok, picture = video.read()
after = np.arange(flight_last + 1, flight_last + 9) - clip_first
after = after[(after < len(b["source"])) & (b["source"][after] == 1)]
for (u, v) in b["position"][after]:
    cv2.circle(picture, (int(u), int(v)), 4, (255, 255, 0), -1)
for (u, v), used in zip(uv_all, keep):
    cv2.circle(picture, (int(u), int(v)), 4, (0, 255, 255) if used else (255, 0, 255), -1)
cv2.polylines(picture, [np.round(cam.project(path3d)).astype(np.int32)], False, (0, 0, 255), 2, cv2.LINE_AA)
ground = path3d * [1, 1, 0]
cv2.polylines(picture, [np.round(cam.project(ground)).astype(np.int32)], False, (255, 255, 255), 1, cv2.LINE_AA)
if bounce is not None:
    (bu, bv), = cam.project([bounce * [1, 1, 0]])
    cv2.circle(picture, (int(bu), int(bv)), 8, (0, 255, 0), 2)
(hu, hv), = cam.project([[hitter[0], hitter[1], 0]])
cv2.circle(picture, (int(hu), int(hv)), 10, (255, 0, 0), 2)
cv2.imwrite("data/flight.png", picture)

# Picture 2: side view (along the court, height), 30 px per metre, net in the middle.
S, W, H = 30, 30 * 34, 30 * 8
side_view = np.full((H, W, 3), (60, 120, 60), np.uint8)
def to_side(y, z):
    return int(W / 2 + y * S), int(H - 10 - z * S)
cv2.line(side_view, to_side(-17, 0), to_side(17, 0), (255, 255, 255), 1)
cv2.line(side_view, to_side(0, 0), to_side(0, 0.914), (30, 30, 30), 3)
for y in (-court.HALF_LENGTH, -court.SERVICE_LINE, court.SERVICE_LINE, court.HALF_LENGTH):
    cv2.line(side_view, to_side(y, 0), to_side(y, -0.2), (255, 255, 255), 2)
cv2.circle(side_view, to_side(hitter[1], 1.0), 6, (255, 0, 0), 2)
pts = np.array([to_side(y, z) for _, y, z in path3d], np.int32)
cv2.polylines(side_view, [pts], False, (0, 0, 255), 2, cv2.LINE_AA)
cv2.imwrite("data/flight_side.png", side_view)
print("Saved data/flight.png and data/flight_side.png")
