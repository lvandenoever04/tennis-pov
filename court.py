import numpy as np

HALF_LENGTH = 11.885           # net to baseline
HALF_WIDTH_DOUBLES = 5.485     # centre to doubles sideline
HALF_WIDTH_SINGLES = 4.115     # centre to singles sideline
SERVICE_LINE = 6.40            # net to service line

L, D, S, V = HALF_LENGTH, HALF_WIDTH_DOUBLES, HALF_WIDTH_SINGLES, SERVICE_LINE

# Every painted line as ((x0, y0), (x1, y1)) in metres.
LINES = [
    ((-D, L), (D, L)),      # far baseline
    ((-D, -L), (D, -L)),    # near baseline
    ((-D, -L), (-D, L)),    # left doubles sideline
    ((D, -L), (D, L)),      # right doubles sideline
    ((-S, -L), (-S, L)),    # left singles sideline
    ((S, -L), (S, L)),      # right singles sideline
    ((-S, V), (S, V)),      # far service line
    ((-S, -V), (S, -V)),    # near service line
    ((0, -V), (0, V)),      # centre service line
]


def points_along_lines(step=0.05):
    """Points every `step` metres along every court line, as an (N, 2) array."""
    points = []
    for (x0, y0), (x1, y1) in LINES:
        n = int(np.hypot(x1 - x0, y1 - y0) / step) + 1
        t = np.linspace(0, 1, n)
        points.append(np.stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t], axis=1))
    return np.concatenate(points)