import sys
import math
import time
import subprocess
import multiprocessing
import cv2
import os
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")      # one maths thread per worker process: we parallelise ourselves
import numpy as np
from numba import njit
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import court
from geometry import Camera

# The whole rally's ball in 3D. Script 13 fits one flight on its own; its weak spot is depth (one camera
# can't see how far away the ball is). Here all flights are fitted together, joined at hits and bounces:
# at a bounce the ball is ON THE GROUND, where the camera does know the depth, and both the incoming and the
# outgoing flight have to pass through that point. So each flight's depth gets pinned by its neighbours.
#
#   1. Find the events (hits and bounces) from the 2D track
#   2. For each stretch between two hits, decide with physics whether (and when) the ball bounced
#   3. Fit all flights together, throw out bad detections, refit
#   4. Ball position for every frame, video, pictures, report

if len(sys.argv) < 2:
    sys.exit("Usage: python 14_ball_3d_rally.py VIDEO [SERVE_KMH ...]\n"
             "   SERVE_KMH: the TV speed board's reading for each serve, in order (optional, for comparison)")
path = sys.argv[1]
board = [float(a) for a in sys.argv[2:]]
stage_start = time.time()


def stage_took(name):
    """Print how long the stage that just finished took."""
    global stage_start
    print(f"  [{name}: {time.time() - stage_start:.1f} s]")
    stage_start = time.time()

# ---- Physics and camera constants (the same as script 13) ----
GRAVITY = 9.81             # m/s^2
DRAG = 0.0204              # 1/m: 0.5 * air density * drag coefficient * area / mass
BALL_RADIUS = 0.033        # m
PIXEL_NOISE = 2.0          # px: typical error of a ball detection
BLUR_ERROR = 0.1           # extra error along the motion, as a fraction of the ball's image movement per frame
OUTLIER = 8.0              # px: a detection this far from the fitted path is a mistake
EXPOSURE = 0.5             # frames: TrackNet's point is the end of each frame's half-frame exposure
CONTACT_AHEAD = 0.5        # m: contact is about this far in front of the player's body...
CONTACT_SPREAD = 0.75      # m: ...give or take this much
REACH = 1.5                # m: beyond this sideways/forwards distance from the body, contact gets expensive
HEIGHT_RANGE = (0.2, 3.3)  # m: lowest and highest contact (3.3: a serve or overhead, arm and racket up)
WALL = 100.0               # px per metre: penalty for the impossible (ball underground, contact too high)
SPIN_RANGE = (-15.0, 6.0)  # m/s^2: Magnus push at the start of a flight (negative = topspin)

# ---- New for joining flights ----
JOIN = 0.05                # m: how closely the end of one flight must meet the start of the next
BOUNCE_HEIGHT = 0.02       # m: how closely a bounce must touch the ground (centre at BALL_RADIUS)
BOUNCE_UP = (0.75, 0.10)   # grass: vertical speed out = -(0.75 +/- 0.10) x vertical speed in
BOUNCE_ALONG = (0.75, 0.15)  # horizontal speed out = (0.75 +/- 0.15) x horizontal speed in
BOUNCE_TURN = 0.10         # rad: how much a bounce may change the horizontal direction
BOUNCE_STEP = 3            # frames: try a bounce every this many frames between two hits...
BOUNCE_FREE = 2.0          # frames: ...each free to move this far...
BOUNCE_NEAR = 4            # frames: ...and only within this far of a break or turn of the 2D track
HIT_WINDOW = 4.0           # frames: a hit may move this far from where the 2D track suggests
TAIL = 5                   # frames of the next flight that are fitted along when judging a stretch (see 2a)
NET_HEIGHT = (0.914, 1.07)  # m: net height in the middle and at the singles sticks...
NET_STICK = court.HALF_WIDTH_SINGLES + 0.914  # ...which stand 0.914 m outside the singles sideline

# ---- Splitting the 2D track into smooth pieces ----
PIECE_FIT = 6.0            # px: a detection further than this from its piece's smooth curve is an outlier
PIECE_COST = 120.0         # cost of starting a new piece (in px^2)...
OUTLIER_COST = 150.0       # ...and of calling a detection an outlier: a break must explain several points
PIECE_LONGEST = 70         # frames
PIECE_GAP = 6              # frames: a piece may not bridge a longer gap (a curve fitted on one side of a long
                           # gap reaches almost anywhere on the other side, and catches false detections there)
MIN_PIECE = 4              # detections: shorter pieces are noise (false detections)

cam = Camera()
b = np.load("data/ball2d.npz")
# Frame numbers inside this script count from the start of the clip (they index the arrays below); the
# report, the video and the saved event frames use the video's own frame numbers (clip_first + frame).
clip_first, fps = int(b["range"][0]), float(b["fps"])
ball2d, source2d = b["position"], b["source"]
n_frames = len(source2d)
players = dict(np.load("data/players.npz"))        # read once (an npz file re-reads an array at every access)
det_frames = np.nonzero(source2d == 1)[0]                  # real detections only, not filled-in ones


def seconds(frame):
    """Clip time of a frame's detection: the END of its exposure. Event 'frames' are on the same clock, so a
    hit at frame 162.5 means contact half way between the detections of frames 162 and 163."""
    return (np.asarray(frame, float) + EXPOSURE) / fps


def shown(frame):
    """The video's frame number for a clip frame, for printing."""
    return frame + clip_first


# ============================== Physics (copied from script 13) ==============================
# Written with plain floats instead of numpy arrays, and compiled to machine code with numba (@njit): the
# fits simulate millions of small steps, which in plain Python took more than half of the running time.

@njit
def acceleration(vx, vy, vz, spin, speed0_sq):
    """Gravity, drag (against the motion, ~speed^2) and Magnus spin push (perpendicular to the motion in the
    vertical plane, ~speed^2; `spin` = its size at the starting speed, negative = topspin = down)."""
    s2 = vx * vx + vy * vy + vz * vz
    s = math.sqrt(s2)
    ux, uy, uz = -vz * vx / s2, -vz * vy / s2, 1.0 - vz * vz / s2      # 'up', made perpendicular to v
    k = spin * s2 / speed0_sq / max(math.sqrt(ux * ux + uy * uy + uz * uz), 1e-9)
    return (-DRAG * s * vx + k * ux, -DRAG * s * vy + k * uy, -GRAVITY - DRAG * s * vz + k * uz)


def simulate(p0, v0, spin, t0, times, step=1 / 200):
    """Positions and velocities at `times` (seconds, any order, before or after t0) of a ball that is at p0
    with velocity v0 at time t0. Times before t0 are found by running the clock backwards."""
    return _simulate(np.asarray(p0, float), np.asarray(v0, float), float(spin), float(t0),
                     np.asarray(times, float), float(step))


@njit
def _simulate(p0, v0, spin, t0, times, step):
    X, V = np.empty((len(times), 3)), np.empty((len(times), 3))
    speed0_sq = max(v0[0] ** 2 + v0[1] ** 2 + v0[2] ** 2, 1e-6)
    for direction in (1, -1):
        sel = np.nonzero((times >= t0) if direction == 1 else (times < t0))[0]
        sel = sel[np.argsort(direction * times[sel])]
        px, py, pz = p0[0], p0[1], p0[2]
        vx, vy, vz = v0[0], v0[1], v0[2]
        now = t0
        for i in sel:
            target = times[i]
            while direction * (target - now) > 1e-9:
                h = direction * min(step, direction * (target - now))
                ax, ay, az = acceleration(vx, vy, vz, spin, speed0_sq)
                mx, my, mz = vx + ax * h / 2, vy + ay * h / 2, vz + az * h / 2   # half a step ahead
                ax, ay, az = acceleration(mx, my, mz, spin, speed0_sq)
                px, py, pz = px + mx * h, py + my * h, pz + mz * h
                vx, vy, vz = vx + ax * h, vy + ay * h, vz + az * h
                now += h
            X[i, 0], X[i, 1], X[i, 2] = px, py, pz
            V[i, 0], V[i, 1], V[i, 2] = vx, vy, vz
    return X, V


simulate([0, 0, 1], [10, 0, 0], 0, 0, [0.1])        # compile now, once, before the work is split over processes


def ray_point(u, v, *, z=None, y=None):
    """The point on the camera ray through pixel (u, v) at height z, or at court position y."""
    ray = cam.R.T @ np.array([(u - cam.cx) / cam.f, (v - cam.cy) / cam.f, 1.0])
    s = (z - cam.C[2]) / ray[2] if z is not None else (y - cam.C[1]) / ray[1]
    return cam.C + s * ray


# ============================== 1. Events from the 2D track ==============================
# A flight projects to a smooth curve in the picture. Cut the track into smooth pieces (a cubic in time for
# each pixel coordinate); every break between pieces is a candidate event. Dynamic programming finds the
# cheapest cut: each piece costs PIECE_COST plus its squared misses, each outlier costs OUTLIER_COST.

def piece_fit(i, j):
    """Fit detections i..j (indices into det_frames) with one smooth curve. Returns (cost, misses)."""
    f = det_frames[i:j + 1].astype(float)
    uv = ball2d[det_frames[i:j + 1]]
    n = len(f)
    if n < 3:
        return n * OUTLIER_COST, None
    A = np.vander(f - f.mean(), 4 if n >= 7 else (3 if n >= 4 else 2))
    use = np.ones(n, bool)
    for _ in range(4):                                 # refit a few times, ignoring points that don't fit
        if use.sum() < A.shape[1] + 1:
            return n * OUTLIER_COST, None
        coef = np.linalg.lstsq(A[use], uv[use], rcond=None)[0]
        miss = np.hypot(*(A @ coef - uv).T)
        use = miss < PIECE_FIT
    if use.sum() >= 2 and np.diff(f[use]).max() > PIECE_GAP:
        return np.inf, None                            # not one piece: split it at the gap
    return np.sum(np.where(use, miss ** 2, OUTLIER_COST)), miss


n_det = len(det_frames)
best, prev = np.full(n_det + 1, np.inf), np.zeros(n_det + 1, int)
best[0] = 0
for j in range(n_det):
    for i in range(j, -1, -1):
        if det_frames[j] - det_frames[i] > PIECE_LONGEST:
            break
        cost = best[i] + PIECE_COST + piece_fit(i, j)[0]
        if cost < best[j + 1]:
            best[j + 1], prev[j + 1] = cost, i
pieces, j = [], n_det
while j > 0:
    pieces.append((prev[j], j - 1))
    j = prev[j]
pieces = pieces[::-1]

# Keep the pieces with enough real points; their inliers are the trustworthy detections.
inliers = []
for i, j in pieces:
    miss = piece_fit(i, j)[1]
    if miss is not None and np.sum(miss < PIECE_FIT) >= MIN_PIECE:
        inliers.append(det_frames[i:j + 1][miss < PIECE_FIT])
print(f"2D track: {n_det} detections, {len(inliers)} smooth pieces, "
      f"{n_det - sum(len(p) for p in inliers)} detections fit no piece (false detections)")


def along_court(u, v):
    """Rough 'how far along the court' for a ball pixel: where its camera ray is 1 m above the ground.
    Not a real position (the ball's height is unknown), but it turns round when someone hits the ball."""
    ray = cam.R.T @ np.array([(u - cam.cx) / cam.f, (v - cam.cy) / cam.f, 1.0])
    return np.nan if ray[2] >= 0 else ray_point(u, v, z=1.0)[1]


# Candidate events: breaks between pieces, and turning points of along_court inside a piece (a hit where
# the ball is nearly still in the picture leaves no visible kink, e.g. a far-side return of serve).
candidates = []                           # (frame before, frame after, pixels nearby)
for pts in inliers:
    f = pts.astype(float)
    A = np.vander(f - f.mean(), 4 if len(f) >= 7 else 3)
    coef = np.linalg.lstsq(A, ball2d[pts], rcond=None)[0]
    fine = np.arange(f[0], f[-1] + 0.01, 0.25)
    curve = np.vander(fine - f.mean(), A.shape[1]) @ coef
    ahead = np.sign(np.diff([along_court(*c) for c in curve]))
    for q in np.nonzero(ahead[1:] * ahead[:-1] < 0)[0]:
        candidates.append((fine[q + 1], fine[q + 1], [curve[q + 1]]))
for before, after in zip(inliers[:-1], inliers[1:]):
    candidates.append((float(before[-1]), float(after[0]), [ball2d[before[-1]], ball2d[after[0]]]))
candidates.sort(key=lambda c: c[0])


def in_racket_zone(frame, who, pixel):
    """Is the pixel within racket reach of the player's box (sideways ~0.6 body heights, above the head)?"""
    x0, y0, x1, y1 = players[f"{who}_box"][int(round(frame))]
    h = y1 - y0
    return x0 - 0.6 * h <= pixel[0] <= x1 + 0.6 * h and y0 - 0.5 * h <= pixel[1] <= y1


# Hits: candidates in exactly one player's racket zone. Players take turns, so when several candidates in
# a row are in the same player's zone, the LAST one is the hit and the earlier ones were something else
# (typically the bounce in front of them, which can be hidden behind their body).
hits = []                                 # [frame before, frame after, who]
for before, after, pixels in candidates:
    who = [w for w in ("near", "far") if any(in_racket_zone((before + after) / 2, w, p) for p in pixels)]
    if len(who) != 1:
        continue
    if hits and hits[-1][2] == who[0]:
        hits[-1] = [before, after, who[0]]
    else:
        hits.append([before, after, who[0]])
rally_start, rally_end = int(inliers[0][0]), int(inliers[-1][-1])
print(f"Ball in play from frame {shown(rally_start)} to {shown(rally_end)}; {len(hits)} hits: "
      + ", ".join(f"{shown((h[0] + h[1]) / 2):.1f} {h[2]}" for h in hits))
stage_took("1. events from the 2D track")


# ============================== 2. Flights joined at events ==============================
# An event: kind ('start', 'hit', 'bounce', 'end'), frame (time), window (lo, hi) it may move in,
# seen (frame where it was seen, or None), who (hitter). Flight m runs from event m to event m+1.

def make_event(kind, frame, lo=None, hi=None, seen=None, who=None, side=None):
    """side: for a bounce that must land on one side of the net, +1 (far, y > 0) or -1 (near, y < 0)."""
    return dict(kind=kind, frame=float(frame), lo=lo, hi=hi, seen=seen, who=who, side=side)


def hitter_position(event):
    return players[f"{event['who']}_position"][int(round(event["frame"]))]


def net_height(x):
    """Height of the top of the net at sideways position x (straight lines from the middle to the sticks)."""
    return NET_HEIGHT[0] + (NET_HEIGHT[1] - NET_HEIGHT[0]) * min(abs(x) / NET_STICK, 1.0)


def net_crossing(t0, p0, t1, times, X):
    """Where a flight from time t0 to t1 crosses the net (y = 0): (x, z), or None. p0 is the position at t0,
    X the positions at `times`; only samples in (t0, t1] count, and between two samples the path is taken as
    straight (they are a frame apart, over which gravity bends the path by about a millimetre)."""
    ps = [p0] + [X[i] for i in np.argsort(times) if t0 < times[i] <= t1]
    for a, c in zip(ps[:-1], ps[1:]):
        if a[1] * c[1] < 0:
            w = a[1] / (a[1] - c[1])
            return a[0] + w * (c[0] - a[0]), a[2] + w * (c[2] - a[2])
    return None


def contact_terms(event, p):
    """Soft prior on a contact point p: a little in front of the hitter, within reach, sensible height."""
    body = hitter_position(event)
    forward = 1.0 if event["who"] == "near" else -1.0
    ahead = (p[1] - body[1] - forward * CONTACT_AHEAD) / CONTACT_SPREAD * PIXEL_NOISE
    reach = max(0.0, math.hypot(p[0] - body[0], p[1] - body[1]) - REACH)
    height = max(0.0, HEIGHT_RANGE[0] - p[2]) + max(0.0, p[2] - HEIGHT_RANGE[1])
    return [ahead, WALL * reach, WALL * height]


class Chain:
    """Several flights joined end to end, fitted together.
    Parameters: per flight its start position (3), start velocity (3) and spin (1); then the times of the
    events that are free to move. Detections are given to the flight whose time span contains them."""

    def __init__(self, events, use):
        self.events, self.use = events, use              # use: frames of the detections to fit
        self.free = [k for k, e in enumerate(events) if e["lo"] is not None]
        self.n_flights = len(events) - 1
        self.memo, self.memo_split = {}, None                 # see residuals()

    def times(self, x):
        T = np.array([e["frame"] for e in self.events])
        T[self.free] = x[7 * self.n_flights:]
        return T

    def assign(self, T):
        """For each flight, given event times T: its detection frames, and all whole frames of its time span
        (used to keep the ball above the grass between detections). Kept fixed while fitting, so the number
        of residuals doesn't change when an event time moves a little."""
        frames = np.asarray(self.use)
        flight = np.clip(np.searchsorted(T, frames, side="right") - 1, 0, self.n_flights - 1)
        # (detections before the first event or after the last still count, judged against the nearest
        # flight: otherwise a fit could 'explain' detections away by moving an event past them)
        return [(frames[flight == m],
                 np.arange(math.floor(T[m]) - 1, math.ceil(T[m + 1]) + 2)) for m in range(self.n_flights)]

    def flight_residuals(self, m, x, T, split):
        """Flight m's own residuals (its detections, the grass, the net), where it ends, and fit details."""
        p, v, spin = x[7 * m:7 * m + 3], x[7 * m + 3:7 * m + 6], x[7 * m + 6]
        frames, between = split[m]
        times = seconds(np.r_[frames, between, T[m + 1]])
        X, V = simulate(p, v, spin, seconds(T[m]), times)
        nd = len(frames)
        # Pixel misses, split into along and across the ball's motion in the picture (blur: see script 13)
        uv = ball2d[frames]
        miss = cam.project(X[:nd]) - uv
        motion = cam.project(X[:nd] + V[:nd] / fps) - cam.project(X[:nd])
        speed = np.maximum(np.linalg.norm(motion, axis=1), 1e-6)
        along = (miss * motion).sum(1) / speed
        across = (miss[:, 0] * motion[:, 1] - miss[:, 1] * motion[:, 0]) / speed
        out = [np.c_[along * PIXEL_NOISE / np.hypot(PIXEL_NOISE, BLUR_ERROR * speed), across].ravel()]
        # under the grass is impossible, but only while this flight is the one in the air
        inside = (times[:-1] > seconds(T[m]) + 0.005) & (times[:-1] < seconds(T[m + 1]) - 0.005)
        out.append(WALL * np.minimum(X[:-1, 2] - BALL_RADIUS, 0.0) * inside)
        # nor can it go through the net: where it crosses, the bottom of the ball must be above the net
        cross = net_crossing(seconds(T[m]), p, seconds(T[m + 1]), times, X)
        out.append([WALL * min(cross[1] - BALL_RADIUS - net_height(cross[0]), 0.0) if cross else 0.0])
        info = dict(frames=frames, miss=np.hypot(*miss.T), across=np.abs(across))
        return out, (X[-1], V[-1]), info

    def residuals(self, x, split=None, details=False):
        T = self.times(x)
        split = self.assign(T) if split is None else split
        # While fitting, most calls change only one or two flights (the fit nudges a few parameters at a time
        # to see what they do), so each flight's residuals are remembered and reused while its parameters and
        # times stay the same.
        memo = self.memo if split is self.memo_split else None
        out, ends, info = [], [], []
        for m in range(self.n_flights):
            key = (m, x[7 * m:7 * m + 7].tobytes(), T[m], T[m + 1])
            if memo is not None and key in memo:
                flight_out, end, flight_info = memo[key]
            else:
                flight_out, end, flight_info = self.flight_residuals(m, x, T, split)
                if memo is not None:
                    memo[key] = flight_out, end, flight_info
            out += flight_out
            ends.append(end)
            info.append(flight_info)
        # Joins between flights, and what each event says about the ball
        for k, e in enumerate(self.events):
            start = x[7 * k:7 * k + 6] if k < self.n_flights else None
            terms = []
            if k > 0 and start is not None:            # the incoming flight must arrive where the next starts
                terms += list((ends[k - 1][0] - start[:3]) / JOIN * PIXEL_NOISE)
            if e["kind"] in ("start", "hit"):
                contact = start[:3] if start is not None else ends[k - 1][0]
                terms += contact_terms(e, contact)
            if e["kind"] == "bounce":
                v_in, v_out = ends[k - 1][1], start[3:6]
                terms.append((start[2] - BALL_RADIUS) / BOUNCE_HEIGHT * PIXEL_NOISE)
                up = -v_out[2] / min(v_in[2], -0.1)
                terms.append((up - BOUNCE_UP[0]) / BOUNCE_UP[1] * PIXEL_NOISE)
                h_in, h_out = np.hypot(*v_in[:2]), np.hypot(*v_out[:2])
                terms.append((h_out / max(h_in, 0.1) - BOUNCE_ALONG[0]) / BOUNCE_ALONG[1] * PIXEL_NOISE)
                turn = math.atan2(v_in[0] * v_out[1] - v_in[1] * v_out[0], v_in[0] * v_out[0] + v_in[1] * v_out[1])
                terms.append(turn / BOUNCE_TURN * PIXEL_NOISE)
                # the rules: a ball that was returned first bounced on the receiver's side of the net
                terms.append(WALL * max(0.0, -e["side"] * start[1]) if e["side"] else 0.0)
            out.append(np.array(terms, float))
        r = np.concatenate(out)
        return (r, info, ends) if details else r

    def sparsity(self, x, split):
        """Which parameters each residual depends on, so the fit only re-simulates what changed."""
        rows, row = [], 0
        for kind, idx, size in self._layout(x, split):
            rows.append((row, size, idx))
            row += size
        S = lil_matrix((row, len(x)), dtype=int)
        for r0, size, idx in rows:
            for c in idx:
                S[r0:r0 + size, c] = 1
        return S

    def _layout(self, x, split):
        """(kind, parameter indices, number of rows) for each block of residuals, in residual order."""
        time_index = {k: 7 * self.n_flights + i for i, k in enumerate(self.free)}
        T = self.times(x)
        layout = []
        for m in range(self.n_flights):
            idx = list(range(7 * m, 7 * m + 7)) + [time_index[k] for k in (m, m + 1) if k in time_index]
            frames, between = split[m]
            layout.append(("pixels", idx, 2 * len(frames)))
            layout.append(("ground", idx, len(frames) + len(between) + 1))       # + 1: the net
        for k, e in enumerate(self.events):
            idx = []
            if k > 0:
                idx += list(range(7 * (k - 1), 7 * k))
            if k < self.n_flights:
                idx += list(range(7 * k, 7 * k + 7))
            idx += [time_index[j] for j in (k - 1, k, k + 1) if j in time_index]
            n = 0
            if k > 0 and k < self.n_flights:
                n += 3
            if e["kind"] in ("start", "hit"):
                n += 3
            if e["kind"] == "bounce":
                n += 5
            layout.append(("event", idx, n))
        return layout

    def bounds(self):
        lo = [-20, -30, 0.0, -80, -80, -80, SPIN_RANGE[0]] * self.n_flights
        hi = [20, 30, 25.0, 80, 80, 80, SPIN_RANGE[1]] * self.n_flights
        lo += [self.events[k]["lo"] for k in self.free]
        hi += [self.events[k]["hi"] for k in self.free]
        return lo, hi

    def initial_guess(self):
        """Event points from the pixels (bounces on the ground, contacts level with the hitter), and for each
        flight the straight-line-plus-gravity velocity from one event point to the next."""
        points = []
        for e in self.events:
            near = np.asarray(self.use)[np.argmin(np.abs(np.asarray(self.use) - e["frame"]))]
            u, v = ball2d[near]
            if e["kind"] == "bounce":
                points.append(np.r_[cam.to_court(u, v), BALL_RADIUS])
            elif e["kind"] in ("start", "hit"):
                forward = 1.0 if e["who"] == "near" else -1.0
                points.append(ray_point(u, v, y=hitter_position(e)[1] + forward * CONTACT_AHEAD))
            else:
                points.append(ray_point(u, v, z=0.5))
        x = []
        for m in range(self.n_flights):
            dt = seconds(self.events[m + 1]["frame"]) - seconds(self.events[m]["frame"])
            v = (points[m + 1] - points[m]) / dt
            v[2] += GRAVITY * dt / 2
            x += list(points[m]) + list(v) + [-2.0]
        x += [self.events[k]["frame"] for k in self.free]
        lo, hi = self.bounds()
        return np.clip(np.array(x, float), np.array(lo) + 1e-6, np.array(hi) - 1e-6)

    def fit(self, x=None, loss="linear", rounds=3):
        """Fit; then give detections to flights by the new event times and fit again, a few times."""
        x = self.initial_guess() if x is None else x
        lo, hi = self.bounds()
        for _ in range(rounds):
            split = self.assign(self.times(x))
            S = self.sparsity(x, split)
            self.memo, self.memo_split = {}, split
            result = least_squares(self.residuals, x, args=(split,), jac_sparsity=S, bounds=(lo, hi),
                                   loss=loss, f_scale=PIXEL_NOISE, x_scale="jac", max_nfev=200)
            same = all(np.array_equal(a[0], c[0]) for a, c in zip(split, self.assign(self.times(result.x))))
            x = result.x
            if same:
                break
        self.memo, self.memo_split = {}, None                 # (don't keep it, nor send it between processes)
        self.x, self.cost = x, result.cost
        return x


# ---- 2a. Did the ball bounce between two hits, and when? Ask the physics. ----
# For each stretch hit -> next hit, try "no bounce" (a volley) and "a bounce at each candidate break", fit
# that little chain, and keep whichever explains the detections best. A bounce hidden behind a player shows
# up as a break with a gap; its time is free within the gap.
# The little chain also carries the first TAIL frames of the NEXT flight. The 2D guess of a hit can be a few
# frames late, and without the next flight the detections between the real hit and the guess would have to
# be explained by the incoming flight, which then 'needs' a made-up bounce to turn round in time.
# Two rules of tennis help: a ball that was returned bounced first on the receiver's side, and no flight
# goes through the net (see Chain.residuals).
use = set(int(f) for p in inliers for f in p)
hit_events = [make_event("hit", (a + c) / 2, (a + c) / 2 - HIT_WINDOW, (a + c) / 2 + HIT_WINDOW, who=w)
              for a, c, w in hits]
start_event = make_event("start", rally_start, who=hits[0][2])
end_event = make_event("end", rally_end)


def bounce_options(lo_frame, hi_frame):
    """Possible bounces between two frames: one every BOUNCE_STEP frames, each free within BOUNCE_FREE, but
    only near a candidate event of the 2D track (a break, a gap or a turn). Away from those the ball is seen
    flying along one smooth curve, so it can't have bounced there. (The breaks are often a frame or two off
    the real bounce, so the physics gets to pick the time.)"""
    return [make_event("bounce", f, f - BOUNCE_FREE, f + BOUNCE_FREE)
            for f in np.arange(lo_frame + 3, hi_frame - 1, BOUNCE_STEP)
            if any(before - BOUNCE_NEAR <= f <= after + BOUNCE_NEAR for before, after, _ in candidates)]


def attempt(task):
    """Fit one stretch, event `first` to event `last` (a hit or the end), with the given bounces.
    Returns (cost, bounces, chain). A plain function of its task, so worker processes can run it."""
    first, last, bounces = task
    tail = [make_event("end", last["frame"] + TAIL)] if last["kind"] == "hit" else []
    frames = sorted(f for f in use if first["frame"] - 0.5 <= f <= (tail or [last])[0]["frame"] + 0.5)
    bounces = [dict(e) for e in bounces]
    if bounces and last["kind"] == "hit":                    # returned: first bounce on the receiver's side
        bounces[0]["side"] = 1 if first["who"] == "near" else -1
    chain = Chain([dict(first)] + bounces + [dict(last)] + tail, frames)
    chain.fit(loss="soft_l1", rounds=2)
    return chain.cost + 5.0 * len(bounces), bounces, chain                  # small price per extra bounce


def run_all(tasks):
    """attempt() for every task, spread over all processor cores; results in the order of the tasks.
    ('fork': the workers start as copies of this process, with everything above already done.)"""
    order = sorted(range(len(tasks)), key=lambda i: -len(tasks[i][2]))    # two-bounce fits (slowest) first
    with multiprocessing.get_context("fork").Pool() as pool:
        done = pool.map(attempt, [tasks[i] for i in order], chunksize=1)
    results = [None] * len(tasks)
    for i, r in zip(order, done):
        results[i] = r
    return results


# Every stretch, with every bounce option, is fitted at the same time. Each stretch starts from the 2D guess
# of its first hit (it may move HIT_WINDOW frames, as before).
stretches = [(first, last) for first, last in zip([start_event] + hit_events, hit_events + [end_event])
             if first["kind"] != "start"]                   # (start -> serve is the toss: up and down in the hand)
options = [[[]] + [[c] for c in bounce_options(first["frame"], last["frame"])] for first, last in stretches]
tasks = [(first, last, o) for (first, last), opts in zip(stretches, options) for o in opts]
results = iter(run_all(tasks))
best = [min((next(results) for _ in opts), key=lambda r: r[0]) for opts in options]
# The last stretch (after the final hit) may have two bounces: try a second one with the first one found.
first, last = stretches[-1]
if best[-1][1]:
    one = best[-1][1][0]
    pairs = [sorted([one, c], key=lambda e: e["frame"]) for c in bounce_options(first["frame"], last["frame"])
             if abs(c["frame"] - one["frame"]) > 5]
    best[-1] = min([best[-1]] + run_all([(first, last, pair) for pair in pairs]), key=lambda r: r[0])

events = [start_event, hit_events[0]]
for (first, last), (cost, bounces, chain) in zip(stretches, best):
    guess = last["frame"]
    # keep the times the little fit found (the bounces', and the end hit's: it saw both sides of that hit),
    # but let them move a bit in the big fit
    times = chain.times(chain.x)
    for e, frame in zip(chain.events, times):
        if e["kind"] == "bounce":
            e["frame"] = float(frame)
    if last["kind"] == "hit":
        last["frame"] = float(times[len(bounces) + 1])
    events += [e for e in chain.events if e["kind"] == "bounce"] + [last]
    found = ", ".join(f"bounce {shown(e['frame']):.1f}" for e in chain.events if e["kind"] == "bounce")
    moved = f" (2D guess {shown(guess):.1f})" if abs(last["frame"] - guess) >= 0.5 else ""
    print(f"  {shown(first['frame']):5.1f} -> {shown(last['frame']):5.1f}{moved}: {found or 'no bounce'}")
stage_took("2. bounce search")
for e in events:                                            # windows for the joint fit
    if e["kind"] == "hit":
        e["lo"], e["hi"] = e["frame"] - 1.5, e["frame"] + 1.5
    elif e["kind"] == "bounce":
        e["lo"], e["hi"] = min(e["lo"], e["frame"] - 0.5), max(e["hi"], e["frame"] + 0.5)

# ============================== 3. Fit everything together ==============================
rally = Chain(events, sorted(use))
x = rally.fit(rounds=2)
for round_ in range(6):                                    # throw out detections that don't fit, refit
    r, info, _ = rally.residuals(x, details=True)
    # a detection is a mistake when it is far off, also compared with how well the rest of its flight fits
    # (a flight into the corners of the picture, like a lob, fits a few pixels worse everywhere: throwing out
    # its detections at 8 px would throw out the very ones that pin down its bounce)
    # All such detections go in one round (one refit per round is what costs the time).
    bad = [f for fl in info if len(fl["frames"])
           for f, miss in zip(fl["frames"], fl["miss"]) if miss > max(OUTLIER, 3 * np.median(fl["miss"]))]
    if not bad:
        break
    rally.use = [f for f in rally.use if f not in bad]
    print(f"  round {round_ + 1}: rejected frames {', '.join(str(int(shown(f))) for f in bad)}")
    x = rally.fit(x, rounds=2)
stage_took("3. joint fit")

# ============================== 4. Results ==============================
T = rally.times(x)
r, info, ends = rally.residuals(x, details=True)
n_flights = rally.n_flights


def seen_bounce(frame):
    """When the 2D track itself says the ball bounced, independent of the 3D fit: fit a smooth curve to the
    detections just before and just after, and find when the two curves meet. None if the bounce wasn't
    in view (fewer than 3 detections on a side, or a gap of more than 2 frames around it)."""
    det = np.array(sorted(rally.use))
    before, after = det[(det > frame - 7) & (det <= frame)], det[(det > frame) & (det <= frame + 7)]
    if len(before) < 3 or len(after) < 3 or after[0] - before[-1] > 2:
        return None
    curves = [np.polyfit(f - frame, ball2d[f], 2) for f in (before, after)]
    fine = np.linspace(before[-1], after[0], 41)
    gap = [np.hypot(*(np.polyval(curves[0], t - frame) - np.polyval(curves[1], t - frame))) for t in fine]
    return float(fine[np.argmin(gap)])


for k, e in enumerate(rally.events):
    if e["kind"] == "bounce":
        e["seen"] = seen_bounce(T[k])


def flight_state(m, frames):
    return simulate(x[7 * m:7 * m + 3], x[7 * m + 3:7 * m + 6], x[7 * m + 6], seconds(T[m]), seconds(frames))


# Every frame: which flight it belongs to, position and velocity at its detection time.
position, velocity = np.full((n_frames, 3), np.nan), np.full((n_frames, 3), np.nan)
SOURCE_UNKNOWN, SOURCE_FITTED, SOURCE_PREDICTED = 0, 1, 2
source = np.zeros(n_frames, int)
flight_of = np.full(n_frames, -1)
used = set(rally.use)
for m in range(n_flights):
    # this flight's frames: from its start event up to, not including, the next event (the last flight
    # includes the end), so a frame just before an event at e.g. 340.5 belongs to the flight that ends there
    frames = np.arange(math.ceil(T[m]), math.floor(T[m + 1]) + 1)
    if m < n_flights - 1:
        frames = frames[frames < T[m + 1]]
    frames = frames[(frames >= 0) & (frames < n_frames)]
    if len(frames) == 0:
        continue
    X, V = flight_state(m, frames)
    position[frames], velocity[frames] = X, V
    flight_of[frames] = m
    source[frames] = [SOURCE_FITTED if f in used else SOURCE_PREDICTED for f in frames]

event_kind = np.array([e["kind"] for e in rally.events])
event_frame = shown(T)                                      # video frame numbers, like the report
event_point = np.array([x[7 * k:7 * k + 3] if k < n_flights else ends[-1][0] for k in range(len(T))])
event_who = np.array([e["who"] or "" for e in rally.events])
np.savez("data/ball3d.npz", range=b["range"], fps=fps, position=position, velocity=velocity, source=source,
         flight=flight_of, event_kind=event_kind, event_frame=event_frame, event_point=event_point,
         event_who=event_who)

# ---- Report ----
print("\nEvents:")
for k, e in enumerate(rally.events):
    p = event_point[k]
    extra = ""
    if e["kind"] in ("hit", "start"):
        extra = f"  by {e['who']} player, {np.hypot(*(p[:2] - hitter_position(dict(e, frame=T[k])))):.2f} m from them"
    if e["kind"] == "bounce":
        inside = abs(p[0]) <= court.HALF_WIDTH_SINGLES and abs(p[1]) <= court.HALF_LENGTH
        seen = f", seen at {shown(e['seen']):.1f}" if e["seen"] is not None else ", hidden"
        extra = f"  {'in' if inside else 'OUT'}{seen}"
    print(f"  frame {shown(T[k]):6.1f}  {e['kind']:6s} at ({p[0]:+6.2f}, {p[1]:+6.2f}, {p[2]:5.2f}) m{extra}")

print("\nFlights:")
print("   frames         hitter  speed     spin   over net    ends            median px  across px")
all_miss, all_across = [], []
for m in range(n_flights):
    first, last = rally.events[m], rally.events[m + 1]
    dense = np.linspace(T[m], T[m + 1], 200)
    X, _ = flight_state(m, dense)
    cross = net_crossing(seconds(T[m]), X[0], seconds(T[m + 1]), seconds(dense), X)
    # gap between the bottom of the ball and the top of the net (negative: through the net)
    net = f"{cross[1] - BALL_RADIUS - net_height(cross[0]):+5.2f} m" if cross else "   -   "
    who = first["who"] if first["kind"] in ("hit", "start") else "(bounce)"
    miss = info[m]["miss"]
    all_miss += list(miss)
    all_across += list(info[m]["across"])
    end = f"{last['kind']} ({event_point[m + 1][0]:+.1f},{event_point[m + 1][1]:+.1f})"
    print(f"  {shown(T[m]):6.1f}-{shown(T[m + 1]):6.1f}  {who:8s} {np.linalg.norm(x[7 * m + 3:7 * m + 6]) * 3.6:4.0f} km/h "
          f"{x[7 * m + 6]:+5.1f}   {net}   {end:22s} "
          f"{np.median(miss) if len(miss) else np.nan:5.1f} ({len(miss):2d})  "
          f"{np.median(info[m]['across']) if len(miss) else np.nan:5.1f}")
print(f"\nOverall: {len(all_miss)} detections used, median {np.median(all_miss):.2f} px "
      f"(across the motion {np.median(all_across):.2f} px), worst {np.max(all_miss):.1f} px")
seen_err = [abs(T[k] - e["seen"]) for k, e in enumerate(rally.events) if e["kind"] == "bounce" and e["seen"] is not None]
if seen_err:
    print(f"Bounce times vs seen: median {np.median(seen_err):.2f} frames, worst {np.max(seen_err):.2f}")
# Serves against the TV speed board. The radar measures the fastest speed, just after contact: compare it
# with the fitted speed at the start of the serve's flight.
serves = [k for k in range(1, n_flights) if rally.events[k - 1]["kind"] == "start"]
for i, k in enumerate(serves):
    fitted = np.linalg.norm(x[7 * k + 3:7 * k + 6]) * 3.6
    reading = (f", speed board {board[i]:.0f} km/h ({board[i] / 1.609:.0f} mph): "
               f"{fitted - board[i]:+.0f} km/h ({(fitted / board[i] - 1) * 100:+.0f}%)" if i < len(board) else "")
    print(f"Serve at frame {shown(T[k]):.1f}: fitted {fitted:.0f} km/h ({fitted / 1.609:.0f} mph){reading}")
print("Frames: " + ", ".join(f"{np.sum(source == s)} {name}" for s, name in
                             ((SOURCE_FITTED, "fitted (detection used)"), (SOURCE_PREDICTED, "predicted by physics"),
                              (SOURCE_UNKNOWN, "unknown"))))

stage_took("4. positions and report")

# ---- Pictures: top-down and side views ----
COLOURS = {"near": (255, 160, 60), "far": (60, 140, 255), "": (200, 200, 200)}   # BGR: near blue, far orange
S_TOP = 25
W_TOP, H_TOP = int(2 * 9 * S_TOP), int(2 * 18 * S_TOP)
top = np.full((H_TOP, W_TOP, 3), (60, 120, 60), np.uint8)
def to_top(px, py):
    return int(W_TOP / 2 + px * S_TOP), int(H_TOP / 2 - py * S_TOP)
for (x0, y0), (x1, y1) in court.LINES:
    cv2.line(top, to_top(x0, y0), to_top(x1, y1), (255, 255, 255), 1)
cv2.line(top, to_top(-6.4, 0), to_top(6.4, 0), (40, 40, 40), 3)
S_SIDE = 25
W_SIDE, H_SIDE = int(36 * S_SIDE), int(10 * S_SIDE)
side = np.full((H_SIDE, W_SIDE, 3), (60, 120, 60), np.uint8)
def to_side(py, pz):
    return int(W_SIDE / 2 + py * S_SIDE), int(H_SIDE - 10 - pz * S_SIDE)
cv2.line(side, to_side(-18, 0), to_side(18, 0), (255, 255, 255), 1)
cv2.line(side, to_side(0, 0), to_side(0, 0.914), (40, 40, 40), 3)
for yy in (-court.HALF_LENGTH, -court.SERVICE_LINE, court.SERVICE_LINE, court.HALF_LENGTH):
    cv2.line(side, to_side(yy, 0), to_side(yy, -0.25), (255, 255, 255), 2)
hitter_of_flight = []
for m in range(n_flights):
    who = ""
    for k in range(m, -1, -1):                              # the last hitter before this flight
        if rally.events[k]["kind"] in ("hit", "start"):
            who = rally.events[k]["who"]
            break
    hitter_of_flight.append(who)
    X, _ = flight_state(m, np.linspace(T[m], T[m + 1], 100))
    cv2.polylines(top, [np.array([to_top(*q[:2]) for q in X], np.int32)], False, COLOURS[who], 1, cv2.LINE_AA)
    cv2.polylines(side, [np.array([to_side(q[1], q[2]) for q in X], np.int32)], False, COLOURS[who], 1, cv2.LINE_AA)
for k, e in enumerate(rally.events):
    p = event_point[k]
    if e["kind"] == "bounce":
        cv2.circle(top, to_top(*p[:2]), 4, (0, 255, 255), -1)
        cv2.circle(side, to_side(p[1], p[2]), 4, (0, 255, 255), -1)
    elif e["kind"] in ("hit", "start"):
        cv2.drawMarker(top, to_top(*p[:2]), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 2)
        cv2.drawMarker(side, to_side(p[1], p[2]), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 10, 2)
cv2.imwrite("data/ball3d_topdown.png", top)
cv2.imwrite("data/ball3d_side.png", side)

# ---- Video: the 3D ball seen by the camera (red), its shadow on the ground (white), events ----
video = cv2.VideoCapture(path)
w, h = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                           "-pix_fmt", "yuv420p", "data/ball3d.mp4"], stdin=subprocess.PIPE)
video.set(cv2.CAP_PROP_POS_FRAMES, clip_first)
for k in range(n_frames):
    ok, frame = video.read()
    if not ok:
        break
    f = clip_first + k
    # events: bounce marks stay on the ground once they've happened; hits show for a moment
    for j, e in enumerate(rally.events):
        if e["kind"] == "bounce" and T[j] <= k:
            (bu, bv), = cam.project([event_point[j] * [1, 1, 0]])
            cv2.ellipse(frame, (int(bu), int(bv)), (9, 4), 0, 0, 360, (0, 255, 0), 2, cv2.LINE_AA)
        if abs(T[j] - k) <= 6 and e["kind"] in ("hit", "bounce"):
            (eu, ev), = cam.project([event_point[j]])
            label = f"HIT ({e['who']})" if e["kind"] == "hit" else "BOUNCE"
            cv2.putText(frame, label, (int(eu) + 15, int(ev) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0) if e["kind"] == "bounce" else (0, 0, 255), 2, cv2.LINE_AA)
    if source[k]:
        trail = [j for j in range(max(0, k - 10), k + 1) if source[j]]
        pts = cam.project(position[trail])
        shadow = cam.project(position[trail] * [1, 1, 0])
        cv2.polylines(frame, [np.round(shadow).astype(np.int32)], False, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.polylines(frame, [np.round(pts).astype(np.int32)], False, (0, 0, 255), 2, cv2.LINE_AA)
        (u, v), (su, sv) = pts[-1], shadow[-1]
        cv2.circle(frame, (int(su), int(sv)), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (int(u), int(v)), 9, (0, 0, 255) if source[k] == SOURCE_FITTED else (255, 0, 255),
                   2, cv2.LINE_AA)
    label = {SOURCE_FITTED: "fitted", SOURCE_PREDICTED: "physics", SOURCE_UNKNOWN: "no ball"}[source[k]]
    height = f"  z {position[k, 2]:.2f} m" if source[k] else ""
    text = f"frame {f}  {label}{height}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
    x0 = (w - tw) // 2                                       # centred, at the top, on a dark box
    cv2.rectangle(frame, (x0 - 12, 18), (x0 + tw + 12, 30 + th + 12), (0, 0, 0), -1)
    cv2.putText(frame, text, (x0, 30 + th), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3, cv2.LINE_AA)
    writer.stdin.write(frame.tobytes())
writer.stdin.close()
writer.wait()
stage_took("5. pictures and video")
print("Saved data/ball3d.npz, data/ball3d.mp4, data/ball3d_topdown.png, data/ball3d_side.png")
