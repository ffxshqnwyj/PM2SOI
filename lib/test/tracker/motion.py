import numpy as np


class BBoxKalmanFilter:
    """Simple constant-velocity Kalman filter for 2D bounding boxes."""

    def __init__(self, dt: float = 1.0, process_var: float = 1e-3, meas_var: float = 1.0):
        self.dt = dt
        self._F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self._F[i, i + 4] = dt
        self._H = np.zeros((4, 8), dtype=np.float32)
        self._H[0, 0] = 1.0
        self._H[1, 1] = 1.0
        self._H[2, 2] = 1.0
        self._H[3, 3] = 1.0
        self._Q = np.eye(8, dtype=np.float32) * process_var
        self._R = np.eye(4, dtype=np.float32) * meas_var
        self._I = np.eye(8, dtype=np.float32)
        self._x = None
        self._P = None
        self._prior = None

    @staticmethod
    def _bbox_to_center(bbox):
        x, y, w, h = bbox
        cx = x + 0.5 * w
        cy = y + 0.5 * h
        return np.array([cx, cy, max(w, 1e-3), max(h, 1e-3)], dtype=np.float32)

    @staticmethod
    def _center_to_bbox(state_vec):
        cx, cy, w, h = state_vec[:4]
        w = max(w, 1e-3)
        h = max(h, 1e-3)
        return np.array([cx - 0.5 * w, cy - 0.5 * h, w, h], dtype=np.float32)

    @property
    def is_initialized(self):
        return self._x is not None

    def reset(self):
        self._x = None
        self._P = None
        self._prior = None

    def init(self, bbox):
        meas = self._bbox_to_center(bbox)
        self._x = np.concatenate([meas, np.zeros(4, dtype=np.float32)], axis=0)
        self._P = np.eye(8, dtype=np.float32)
        self._prior = self._x.copy()

    def predict(self):
        if not self.is_initialized:
            return None
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        self._prior = self._x.copy()
        return self.get_prediction_bbox()

    def update(self, bbox):
        if bbox is None:
            return
        if not self.is_initialized:
            self.init(bbox)
            return
        z = self._bbox_to_center(bbox)
        y = z - (self._H @ self._x)
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        self._P = (self._I - K @ self._H) @ self._P

    def get_prediction_bbox(self):
        if self._prior is None:
            return None
        return self._center_to_bbox(self._prior)

    def get_state_bbox(self):
        if not self.is_initialized:
            return None
        return self._center_to_bbox(self._x)
