import os
import sys
import time
import warnings
import subprocess
import cv2
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import spsolve
import court
from geometry import Camera

if len(sys.argv) < 2:
    sys.exit("Usage: python 16_head_direction.py VIDEO [WEIGHTS]   e.g. python 16_head_direction.py data/centre25.mp4\n"
             "   WEIGHTS defaults to ~/projects/DirectMHP/weights/agora_m_1280_e300_t40_lw010_best.pt")
path = sys.argv[1]
weights = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else
                             "~/projects/DirectMHP/weights/agora_m_1280_e300_t40_lw010_best.pt")

# Use the DirectMHP repo's own non-maximum suppression. DirectMHP is a YOLOv5 head detector whose detections
# carry three extra numbers: the head's pitch, yaw and roll, relative to the camera.
sys.path.insert(0, os.path.expanduser("~/projects/DirectMHP"))
warnings.filterwarnings("ignore", category=UserWarning)     # its old code imports the deprecated pkg_resources
from utils.general import non_max_suppression              # type: ignore

SCALES = (0.5, 0.6, 0.67)     # the frame is shrunk to these sizes (tried 0.5-1.5 and crops: these find the most
                              # heads), and each is also run mirrored: 6 'views' of every frame
MIN_DETECTION = 0.1           # detector confidence kept; detections are then matched to our two players
HEAD_SIZE = 0.25              # m: a head is about this big
MATCH = 1.2                   # a detection must be within this many head sizes of our head position (script 15)
CONFIDENT = 0.5               # quality (see below) at which a measurement counts as confident in the report
FACING_NET = {"near": 0.0, "far": 180.0}   # court yaw of a player looking straight at the opponent
MAX_LAG = 0.4                 # s: longest delay tried for "the head follows the ball"
NEUTRAL = 60.0                # deg: with nothing else to go on, a player faces the net (very weak pull)
TURN_ACCEL = 2.0              # deg/frame^2: how quickly a head can change its turning speed (smoothness)
MAX_TURN_SPEED = 500.0        # deg/s: fastest a head turns (quick head turns in sport peak at about 400-600)
ARROW = 2.0                   # m: length of the gaze arrow drawn in the video
COLOURS = {"near": (60, 60, 230), "far": (230, 150, 50)}   # BGR, as in script 15
MEASURED, BALL, SMOOTHED = 1, 2, 0                          # where each frame's final direction came from

start_time = stage_start = time.time()
def stage_took(name):
    """Print how long the stage that just finished took."""
    global stage_start
    print(f"  [{name}: {time.time() - stage_start:.1f} s]")
    stage_start = time.time()


cam = Camera()
heads = np.load("data/heads.npz")
ball3d = np.load("data/ball3d.npz")
first, last = (int(v) for v in heads["range"])
fps = float(heads["fps"])
n = last - first + 1
head = {name: heads[f"{name}_head"] for name in FACING_NET}
ball = ball3d["position"]                      # NaN where the ball isn't known (before the serve, after the end)

# ---- 1. Measure: run DirectMHP on every frame ----
device = "cuda" if torch.cuda.is_available() else "cpu"
# The file is a whole pickled model (YOLOv5 style), not just weights, so it can't be loaded weights-only.
checkpoint = torch.load(weights, map_location=device, weights_only=False)
model = (checkpoint.get("ema") or checkpoint["model"]).float().fuse().eval().to(device)


def detect(frame, scale):
    """All heads DirectMHP finds in the frame shrunk by `scale`, as-is and mirrored. For each of the two:
    (centre u, centre v, confidence, yaw, pitch), in full-frame pixels and un-mirrored angles (degrees)."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (round(w * scale), round(h * scale)))
    sh, sw = small.shape[:2]
    H, W = int(np.ceil(sh / 64) * 64), int(np.ceil(sw / 64) * 64)     # the network wants multiples of 64
    batch = np.full((2, H, W, 3), 114, np.uint8)                       # grey padding, as in YOLOv5
    batch[0, :sh, :sw], batch[1, :sh, :sw] = small, small[:, ::-1]
    x = torch.from_numpy(batch[..., ::-1].transpose(0, 3, 1, 2).copy()).to(device).float() / 255   # RGB, 0-1
    with torch.no_grad():
        found = non_max_suppression(model(x)[0], MIN_DETECTION, 0.45, num_angles=3)
    views = []
    for mirrored, d in enumerate(found):
        d = d.cpu().numpy()
        u, v = (d[:, 0] + d[:, 2]) / 2, (d[:, 1] + d[:, 3]) / 2
        if mirrored:                       # mirroring flips left/right: undo it for the position and the yaw
            u = sw - u
        yaw = (d[:, 7] - 0.5) * 360 * (-1 if mirrored else 1)          # the network outputs angles as 0-1
        pitch = (d[:, 6] - 0.5) * 180
        views.append((u / scale, v / scale, d[:, 4], yaw, pitch))
    return views


N_VIEWS = 2 * len(SCALES)
raw = {name: np.full((n, N_VIEWS, 3), np.nan) for name in FACING_NET}   # per view: confidence, yaw, pitch
video = cv2.VideoCapture(path)
video.set(cv2.CAP_PROP_POS_FRAMES, first)
for k in range(n):
    ok, frame = video.read()
    if not ok:
        break
    views = [view for s in SCALES for view in detect(frame, s)]
    for name in FACING_NET:
        (hu, hv), = cam.project([head[name][k]])
        depth = (cam.R @ (head[name][k] - cam.C))[2]
        radius = MATCH * cam.f * HEAD_SIZE / depth                     # pixels
        for i, (u, v, conf, yaw, pitch) in enumerate(views):
            near_enough = np.hypot(u - hu, v - hv) < radius
            if near_enough.any():
                j = np.flatnonzero(near_enough)[np.argmax(conf[near_enough])]   # the most confident one
                raw[name][k, i] = conf[j], yaw[j], pitch[j]
    if k % 100 == 0:
        print(f"  frame {first + k}")
stage_took("1. DirectMHP on every frame")

# ---- 2. Convert to court directions and evaluate against "towards the ball" ----


def camera_direction(yaw, pitch):
    """DirectMHP's angles -> the direction the face points, in camera coordinates (x right, y down, z away
    from the camera). Yaw 0, pitch 0 is a face looking straight at the camera; yaw 180 is the back of a head.
    The formula is the 'Z axis' DirectMHP draws in its own demos (utils/plots.py, plot_3axis_Zaxis), whose
    image part is (-sin yaw, -sin pitch cos yaw); the depth part follows from it being a unit vector."""
    y, p = np.radians(yaw), np.radians(pitch)
    return np.stack([-np.sin(y), -np.sin(p) * np.cos(y), -np.cos(p) * np.cos(y)], axis=-1)


def line_of_sight_turn(point):
    """Rotation (camera coordinates) that turns the camera's axis onto the line of sight to `point`."""
    ray = cam.R @ (point - cam.C)
    ray /= np.linalg.norm(ray)
    axis = np.cross([0, 0, 1], ray)
    angle = np.arcsin(np.linalg.norm(axis))
    return cv2.Rodrigues(axis / np.linalg.norm(axis) * angle)[0]


def angles(d):
    """Court direction(s) -> (yaw, pitch) in degrees. Yaw 0 points at the far baseline (+y), 90 to +x;
    pitch is up from horizontal."""
    return np.degrees(np.arctan2(d[..., 0], d[..., 1])), np.degrees(np.arcsin(np.clip(d[..., 2], -1, 1)))


def wrap(a):
    """Angle(s) in degrees -> the same angle between -180 and 180."""
    return (a + 180) % 360 - 180


def towards_ball(name, lag):
    """Unit direction from the head to where the ball was `lag` frames ago (NaN where unknown)."""
    earlier = np.clip(np.arange(n) - lag, 0, n - 1)
    d = ball[earlier] - head[name]
    d[np.arange(n) < lag] = np.nan
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def angle_between(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1, 1)))


measured, quality, spread = {}, {}, {}
both_ways = {}                                   # median error vs the ball, for the two ways of reading angles
for name in FACING_NET:
    conf, yaw, pitch = raw[name][..., 0], raw[name][..., 1], raw[name][..., 2]
    d_cam = camera_direction(yaw, pitch)                                       # (n, views, 3)
    # Angles relative to the camera's axis, or relative to the line of sight to the head? The network only
    # sees how the head looks, which depends on the line of sight, but it was trained on labels relative to
    # the camera axis. The heads here are up to 14 degrees off-axis. Both are compared with the ball in the
    # report: on this rally the line of sight is 4 degrees better for the far player (17 vs 21) and 2 worse
    # for the near one (20 vs 17), so better overall, and it is the physically right reading of a picture.
    turned = np.einsum("kij,kvj->kvi", np.array([line_of_sight_turn(p) for p in head[name]]), d_cam)
    for way, d in (("camera axis", d_cam), ("line of sight", turned)):
        d_court = d @ cam.R                                                    # camera -> court: R transposed
        # Combine the views: average the directions, weighted by the detector's confidence.
        w = np.nan_to_num(conf)
        total = np.einsum("kv,kvi->ki", w, np.nan_to_num(d_court))
        length = np.linalg.norm(total, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            direction = total / length[:, None]
            agreement = length / w.sum(axis=1)  # 1: all views point the same way; lower: they disagree
        # Quality 0-1: the average confidence over all 6 views (a view that found nothing counts as 0),
        # times how well the views agree. A head found in every view, all pointing the same way, is near 1.
        q = np.nan_to_num(w.sum(axis=1) / N_VIEWS * agreement)
        if way == "line of sight":
            measured[name], quality[name] = direction, q
            off = angle_between(d_court, direction[:, None])            # each view vs the average
            spread[name] = np.nansum(off, axis=1) / np.maximum(np.sum(~np.isnan(off), axis=1), 1)
        both_ways.setdefault(name, {})[way] = direction

# How long after the ball does the head follow it? Try every lag up to MAX_LAG, using the confident
# measurements of both players, and keep the one where the head points most closely at the ball.
lags = range(int(MAX_LAG * fps) + 1)
lag_error = []
for lag in lags:
    errors = [angle_between(measured[name], towards_ball(name, lag))[quality[name] >= CONFIDENT] for name in FACING_NET]
    lag_error.append(np.nanmedian(np.concatenate(errors)))
LAG = int(np.argmin(lag_error))
print(f"\nHead vs ball, median angle on confident frames, by delay: "
      + ", ".join(f"{lag / fps:.2f} s {e:.1f}" for lag, e in zip(lags, lag_error)))
print(f"  -> the head follows the ball {LAG / fps:.2f} s ({LAG} frames) late")

print("\nMeasurement (DirectMHP) vs 'towards the ball':")
noise, prior_sigma = {}, {}
for name in FACING_NET:
    who = {"near": "Alcaraz, near, seen from behind", "far": "Fognini, far, seen from the front"}[name]
    q, d = quality[name], measured[name]
    found, sure = q > 0, q >= CONFIDENT
    b = towards_ball(name, LAG)
    yaw_m, pitch_m = angles(d)
    yaw_b, pitch_b = angles(b)
    err = angle_between(d, b)
    has_ball = ~np.isnan(err)
    dyaw, dpitch = wrap(yaw_m - yaw_b), pitch_m - pitch_b
    # frame-to-frame noise: the change between consecutive confident frames. A real head moves only a few
    # degrees per frame, so most of this is noise; the robust spread (1.48 x median absolute value)
    # divided by sqrt(2) (each change contains two noisy measurements) estimates the noise of one frame.
    pair = sure[1:] & sure[:-1]
    step_yaw = wrap(np.diff(yaw_m))[pair]
    step_pitch = np.diff(pitch_m)[pair]
    noise[name] = (1.48 * np.median(np.abs(step_yaw)) / np.sqrt(2), 1.48 * np.median(np.abs(step_pitch)) / np.sqrt(2))
    # how far players look from the ball, beyond measurement noise: the spread of (measured - ball) on
    # confident frames, with the measurement noise taken out (they add up as squares)
    use = sure & has_ball
    total_yaw, total_pitch = 1.48 * np.median(np.abs(dyaw[use])), 1.48 * np.median(np.abs(dpitch[use] - np.median(dpitch[use])))
    prior_sigma[name] = (np.sqrt(max(total_yaw ** 2 - noise[name][0] ** 2, 0)),
                         np.sqrt(max(total_pitch ** 2 - noise[name][1] ** 2, 0)))
    print(f"\n  {name} player ({who}):")
    print(f"    head found in {found.sum()} of {n} frames, confidently (quality >= {CONFIDENT}) in {sure.sum()}; "
          f"median quality {np.median(q[found]):.2f}, views disagree by {np.nanmedian(spread[name]):.0f} deg on median")
    for label, m in (("all found", found & has_ball), ("confident", sure & has_ball)):
        print(f"    vs ball ({label}, {m.sum()} frames): median {np.median(err[m]):.0f} deg, "
              f"within 30 deg {np.mean(err[m] < 30) * 100:.0f}%, more than 90 deg off {np.mean(err[m] > 90) * 100:.0f}%; "
              f"yaw median |error| {np.median(np.abs(dyaw[m])):.0f} deg, pitch bias {np.median(dpitch[m]):+.0f} deg "
              f"(|error| {np.median(np.abs(dpitch[m])):.0f})")
    print(f"    measured yaw follows the ball's yaw: correlation {np.corrcoef(yaw_m[use], yaw_b[use])[0, 1]:+.2f} "
          f"(range of ball yaw {np.percentile(wrap(yaw_b[use] - FACING_NET[name]), 5):+.0f} to "
          f"{np.percentile(wrap(yaw_b[use] - FACING_NET[name]), 95):+.0f} deg from facing the net)")
    print(f"    frame to frame (confident neighbours): yaw changes median {np.median(np.abs(step_yaw)):.1f} deg, "
          f"90th percentile {np.percentile(np.abs(step_yaw), 90):.1f}; pitch median {np.median(np.abs(step_pitch)):.1f} deg "
          f"-> noise of one frame about {noise[name][0]:.1f} deg yaw, {noise[name][1]:.1f} deg pitch")
    print(f"    angles relative to camera axis vs line of sight: median error "
          + ", ".join(f"{way} {np.nanmedian(angle_between(dd, b)[use]):.1f} deg" for way, dd in both_ways[name].items()))
    print(f"    looking away from the ball beyond the noise: about {prior_sigma[name][0]:.0f} deg yaw")
    far_ball = use & (np.linalg.norm(ball - head[name], axis=1) > 3)
    print(f"    pitch, ball more than 3 m away: measured = {np.polyfit(pitch_b[far_ball], pitch_m[far_ball], 1)[0]:.2f} "
          f"x ball pitch {np.median(dpitch[far_ball]):+.0f} deg (1.00 x +0 would be a head that follows the ball)")
stage_took("2. evaluation")

# ---- 3. Combine into one smooth direction per player ----
# Yaw only from DirectMHP: its pitch hardly follows the ball (on confident frames with the ball more than 3 m
# away, measured pitch = 0.1-0.25 x ball pitch) and sits at about -10 deg for the near player and +11 for
# the far one whatever happens: that is just "level with the camera" (which looks down at them by about
# 8-14 degrees) turned into court coordinates. These heads are 15-40 pixels, too small to see pitch.
# So pitch comes from the ball and smoothness alone.
# For each angle (yaw, pitch), find the path theta(t) that minimises
#     sum over frames of  quality * ((theta - measured) / noise)^2          the measurement, if any
#                       + ((theta - towards ball, LAG frames ago) / look_away)^2      the ball prior, if known
#                       + ((theta - facing the net) / NEUTRAL)^2             a very weak default
#                       + ((change in turning speed) / TURN_ACCEL)^2         smooth, like a real head
# Each term is "how many standard deviations off", squared, so the weights come straight from the measured
# spreads in the evaluation above: noise = measurement noise of that player, look_away = how far that player
# looks from the ball beyond the noise. A measurement with quality 1 then counts as much as one frame of
# noise-level evidence; a doubtful one proportionally less. It is a linear least-squares problem: one sparse
# solve. Measurements (and ball prior) that end up far from the result are down-weighted and it is solved
# again (3 rounds), so that occasional front/back mix-ups of the network can't pull the head round.
MIN_SIGMA = 5.0          # deg: no estimate is trusted to better than this (avoids near-zero spreads)
second_difference = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
smoothness = (second_difference.T @ second_difference) / TURN_ACCEL ** 2


def fuse(meas, w_meas, sigma_meas, prior, sigma_prior, neutral, circular):
    """The minimisation above for one angle. Returns the path and the final (down-weighted) weights."""
    base_m = np.where(np.isnan(meas), 0, w_meas) / sigma_meas ** 2
    base_p = np.where(np.isnan(prior), 0, 1.0) / sigma_prior ** 2
    theta = np.full(n, neutral)
    wm, wp = base_m, base_p
    for _ in range(3):
        # express every measurement as the nearest equivalent angle to the current path (yaw: -170 and 190
        # are the same direction)
        m = theta + (wrap(meas - theta) if circular else meas - theta)
        p = theta + (wrap(prior - theta) if circular else prior - theta)
        A = sparse.diags(wm + wp + 1 / NEUTRAL ** 2) + smoothness
        rhs = wm * np.nan_to_num(m) + wp * np.nan_to_num(p) + neutral / NEUTRAL ** 2
        theta = spsolve(A.tocsc(), rhs)
        # down-weight what is far off: 3 standard deviations off keeps half its weight, 6 keeps a fifth
        wm = base_m / (1 + (np.nan_to_num(m - theta) / (3 * sigma_meas)) ** 2)
        wp = base_p / (1 + (np.nan_to_num(p - theta) / (3 * sigma_prior)) ** 2)
    return theta, wm, wp


def limit_turn_speed(yaw, pitch):
    """Cap how far the head turns from one frame to the next (both angles together). Returns how many frames
    were capped."""
    max_step = MAX_TURN_SPEED / fps
    capped = 0
    for k in range(1, n):
        dy, dp = (yaw[k] - yaw[k - 1]) * np.cos(np.radians(pitch[k])), pitch[k] - pitch[k - 1]
        step = np.hypot(dy, dp)
        if step > max_step:
            yaw[k] = yaw[k - 1] + (yaw[k] - yaw[k - 1]) * max_step / step
            pitch[k] = pitch[k - 1] + dp * max_step / step
            capped += 1
    return capped


results = {}
print("\nCombined head direction:")
for name in FACING_NET:
    yaw_m, pitch_m = angles(measured[name])
    yaw_b, pitch_b = angles(towards_ball(name, LAG))
    q = np.where(np.isnan(yaw_m), 0, quality[name])
    # yaw relative to facing the net, so both players' paths are near 0 and never cross +-180
    rel_m, rel_b = wrap(yaw_m - FACING_NET[name]), wrap(yaw_b - FACING_NET[name])
    s_noise = [max(s, MIN_SIGMA) for s in noise[name]]
    s_prior = [max(s, MIN_SIGMA) for s in prior_sigma[name]]
    rel, wm_yaw, wp_yaw = fuse(rel_m, q, s_noise[0], rel_b, s_prior[0], 0.0, circular=True)
    pitch, _, _ = fuse(pitch_m, np.zeros(n), s_noise[1], pitch_b, s_prior[0], 0.0, circular=False)   # see above
    capped = limit_turn_speed(rel, pitch)
    yaw = wrap(rel + FACING_NET[name])
    y, p = np.radians(yaw), np.radians(pitch)
    forward = np.stack([np.sin(y) * np.cos(p), np.cos(y) * np.cos(p), np.sin(p)], axis=1)
    # which evidence carried each frame's yaw (pitch always comes from the ball, see above)
    source = np.where((wm_yaw > 0) & (wm_yaw >= wp_yaw), MEASURED, np.where(wp_yaw > 0, BALL, SMOOTHED))
    results[name] = dict(forward=forward, yaw=yaw, pitch=pitch, source=source,
                         measured_share=wm_yaw / (wm_yaw + wp_yaw + 1e-12))

    speed = np.hypot(np.diff(rel) * np.cos(p[1:]), np.diff(pitch)) * fps
    err = angle_between(forward, towards_ball(name, LAG))
    err_now = angle_between(forward, towards_ball(name, 0))
    print(f"  {name}: weights: measurement noise {s_noise[0]:.1f} deg yaw, looking away from the ball "
          f"{s_prior[0]:.1f} deg (yaw and pitch; pitch not measured)")
    print(f"    yaw led by: measurement {np.sum(source == MEASURED)}, ball {np.sum(source == BALL)}, "
          f"smoothing only {np.sum(source == SMOOTHED)}; measurement's average share of the evidence "
          f"{np.mean(results[name]['measured_share']) * 100:.0f}%")
    print(f"    vs ball: median {np.nanmedian(err):.0f} deg ({LAG / fps:.2f} s ago), {np.nanmedian(err_now):.0f} deg "
          f"(now); yaw {np.percentile(wrap(yaw - FACING_NET[name]), 2):+.0f} to "
          f"{np.percentile(wrap(yaw - FACING_NET[name]), 98):+.0f} deg from facing the net, "
          f"pitch {np.percentile(pitch, 2):+.0f} to {np.percentile(pitch, 98):+.0f}")
    print(f"    turning speed: median {np.median(speed):.0f} deg/s, 98th percentile {np.percentile(speed, 98):.0f}, "
          f"top {speed.max():.0f} (frame {first + 1 + int(np.argmax(speed))}); capped at {MAX_TURN_SPEED:.0f} "
          f"in {capped} frames; frame to frame median change {np.median(speed) / fps:.1f} deg")

np.savez("data/head_direction.npz", range=heads["range"], fps=fps, lag_frames=LAG,
         **{f"{name}_{key}": r[key] for name, r in results.items() for key in ("forward", "yaw", "pitch", "source")},
         **{f"{name}_measured": measured[name] for name in FACING_NET},
         **{f"{name}_quality": quality[name] for name in FACING_NET})
print("\nSaved data/head_direction.npz  (per player: forward (unit vector, court coordinates), yaw (deg, 0 = towards "
      "the far baseline, 90 = towards +x), pitch (deg, up, from the ball), source of the yaw (1 measured, 2 ball, 0 smoothing only), "
      "plus the raw measurement and its quality)")
stage_took("3. combine")

# ---- 4. Picture: yaw over the rally (dots: measurement, darker = higher quality; thin: ball; thick: result) ----
W, H_ONE = max(n, 700) * 2, 300
graph = np.full((2 * H_ONE, W, 3), 30, np.uint8)
for row, name in enumerate(FACING_NET):
    def gy(a):
        return int(row * H_ONE + H_ONE / 2 - a * (H_ONE - 30) / 2 / 120)          # +-120 deg fills the panel
    def gx(k):
        return int(k * W / n)
    for a in (-90, -45, 0, 45, 90):
        cv2.line(graph, (0, gy(a)), (W, gy(a)), (70, 70, 70), 1)
        cv2.putText(graph, f"{a:+d}", (4, gy(a) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
    yaw_m = wrap(angles(measured[name])[0] - FACING_NET[name])
    for k in np.flatnonzero(quality[name] > 0):
        shade = int(80 + 175 * min(quality[name][k], 1))
        cv2.circle(graph, (gx(k), gy(yaw_m[k])), 2, (shade, shade, shade), -1)
    yaw_b = wrap(angles(towards_ball(name, LAG))[0] - FACING_NET[name])
    for k in range(n - 1):
        if not np.isnan(yaw_b[k]) and not np.isnan(yaw_b[k + 1]) and abs(yaw_b[k + 1] - yaw_b[k]) < 90:
            cv2.line(graph, (gx(k), gy(yaw_b[k])), (gx(k + 1), gy(yaw_b[k + 1])), (0, 200, 255), 1, cv2.LINE_AA)
    final = wrap(results[name]["yaw"] - FACING_NET[name])
    pts = np.array([(gx(k), gy(a)) for k, a in enumerate(final)], np.int32)
    cv2.polylines(graph, [pts], False, COLOURS[name], 2, cv2.LINE_AA)
    cv2.putText(graph, f"{name} player: yaw from facing the net (deg). dots: DirectMHP (brighter = surer), "
                f"yellow: towards the ball {LAG / fps:.2f} s ago, thick: combined",
                (60, row * H_ONE + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    for f in range(0, n, 100):
        cv2.putText(graph, str(first + f), (gx(f) + 2, (row + 1) * H_ONE - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (160, 160, 160), 1)
cv2.imwrite("data/head_direction.png", graph)
print("Saved data/head_direction.png")

# ---- 5. Video: gaze arrows from each head, and a top-down view in the corner ----
# Thick arrow: the combined direction (2 m long). Thin white arrow: the raw DirectMHP measurement, when found.
# Top-down view: players' gaze lines and the ball, as seen from above (near baseline at the bottom).
S_TOP = 12                                     # pixels per metre in the top-down view
W_TOP, H_TOP = int(16 * S_TOP), int(32 * S_TOP)
def to_top(x, y):
    return int(W_TOP / 2 + x * S_TOP), int(H_TOP / 2 - y * S_TOP)
top_court = np.full((H_TOP, W_TOP, 3), (60, 120, 60), np.uint8)
for (x0, y0), (x1, y1) in court.LINES:
    cv2.line(top_court, to_top(x0, y0), to_top(x1, y1), (255, 255, 255), 1)
cv2.line(top_court, to_top(-6.4, 0), to_top(6.4, 0), (40, 40, 40), 2)

how = {MEASURED: "measured", BALL: "ball", SMOOTHED: "smoothed"}
video = cv2.VideoCapture(path)
video.set(cv2.CAP_PROP_POS_FRAMES, first)
w, h = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                           "-pix_fmt", "yuv420p", "data/head_direction.mp4"], stdin=subprocess.PIPE)
for k in range(n):
    ok, frame = video.read()
    if not ok:
        break
    top = top_court.copy()
    for name in FACING_NET:
        r, eye = results[name], head[name][k]
        if quality[name][k] > 0:
            (u0, v0), (u1, v1) = cam.project([eye, eye + ARROW * measured[name][k]])
            cv2.arrowedLine(frame, (int(u0), int(v0)), (int(u1), int(v1)), (255, 255, 255), 1, cv2.LINE_AA,
                            tipLength=0.15)
        (u0, v0), (u1, v1) = cam.project([eye, eye + ARROW * r["forward"][k]])
        cv2.arrowedLine(frame, (int(u0), int(v0)), (int(u1), int(v1)), COLOURS[name], 3, cv2.LINE_AA, tipLength=0.2)
        label = f"yaw {r['yaw'][k]:+.0f} pitch {r['pitch'][k]:+.0f} ({how[r['source'][k]]})"
        for colour, thick in (((0, 0, 0), 4), (COLOURS[name], 2)):
            cv2.putText(frame, label, (int(u0) + 14, int(v0) - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, thick)
        cv2.circle(top, to_top(*eye[:2]), 4, COLOURS[name], -1, cv2.LINE_AA)
        g = r["forward"][k][:2] / max(np.linalg.norm(r["forward"][k][:2]), 1e-6)
        cv2.line(top, to_top(*eye[:2]), to_top(*(eye[:2] + 6 * g)), COLOURS[name], 2, cv2.LINE_AA)
    if not np.isnan(ball[k, 0]):
        cv2.circle(top, to_top(*ball[k, :2]), 3, (0, 255, 255), -1, cv2.LINE_AA)
    frame[90:90 + H_TOP, 10:10 + W_TOP] = top
    text = f"frame {first + k}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
    x0 = (w - tw) // 2                                       # centred, at the top, on a dark box
    cv2.rectangle(frame, (x0 - 12, 18), (x0 + tw + 12, 30 + th + 12), (0, 0, 0), -1)
    cv2.putText(frame, text, (x0, 30 + th), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    writer.stdin.write(frame.tobytes())
writer.stdin.close()
writer.wait()
print("Saved data/head_direction.mp4")
stage_took("4-5. picture and video")
print(f"\nTotal {time.time() - start_time:.0f} s")
