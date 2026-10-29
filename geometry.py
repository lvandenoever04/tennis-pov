import numpy as np


class Camera:
    """The broadcast camera from scripts 7 and 8: converts between pixels and court positions (metres)."""

    def __init__(self, camera_file="data/camera.npz", court_file="data/court_track.npz"):
        cam = np.load(camera_file)
        self.f, self.R, self.C = float(cam["f"]), cam["R"], cam["C"]
        self.cx, self.cy = float(cam["cx"]), float(cam["cy"])
        self.H = np.load(court_file)["homographies"][0]
        self.H_inverse = np.linalg.inv(self.H)

    def to_court(self, u, v):
        """A pixel on the ground -> court position (x across, y along) in metres."""
        p = self.H_inverse @ np.array([u, v, 1.0])
        return p[:2] / p[2]

    def height_above_ground(self, u, v, gx, gy):
        """How high above the ground point (gx, gy) the pixel (u, v) is, in metres."""
        ray = self.R.T @ np.array([(u - self.cx) / self.f, (v - self.cy) / self.f, 1.0])
        t = ((gx - self.C[0]) * ray[0] + (gy - self.C[1]) * ray[1]) / (ray[0] ** 2 + ray[1] ** 2)
        return self.C[2] + t * ray[2]

    def project(self, points3d):
        """3D points (metres) -> pixels."""
        K = np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1.0]])
        x = (K @ (self.R @ (np.asarray(points3d, float).reshape(-1, 3) - self.C).T)).T
        return x[:, :2] / x[:, 2:3]