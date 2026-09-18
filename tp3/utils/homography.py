import math
import cv2
import numpy as np
import config


class HomographyManager:

    def __init__(self) -> None:
        self._H: np.ndarray | None = None
        self._H_inv: np.ndarray | None = None
        self.metric_bounds: tuple[float, float, float, float] | None = None
        self.zenithal_background: np.ndarray | None = None

    def compute_homography(
        self,
        pixel_points: np.ndarray,
        metric_points: np.ndarray,
    ) -> np.ndarray:
        n = len(pixel_points)
        if n < 4:
            raise ValueError(f"Need at least 4 point correspondences; got {n}.")
        src = pixel_points.astype(np.float32)
        dst = metric_points.astype(np.float32)
        if n == 4:
            H = cv2.getPerspectiveTransform(src, dst)
        else:
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=5.0)
            if H is None:
                raise RuntimeError("cv2.findHomography failed to compute H.")
        self._H = H
        self._H_inv = np.linalg.inv(H)
        xs, ys = metric_points[:, 0], metric_points[:, 1]
        self.metric_bounds = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        return H

    def capture_zenithal_background(
        self,
        frame: np.ndarray,
        output_size: tuple[int, int] = config.W2D_SIZE,
    ) -> np.ndarray:
        half_extent = config.W2D_HALF_EXTENT_MM
        bounds = (-half_extent, -half_extent, half_extent, half_extent)
        self.zenithal_background = self.warp_to_zenithal(frame, output_size, bounds)
        return self.zenithal_background.copy()

    def pixel_to_metric(self, pixel_pts: np.ndarray) -> np.ndarray:
        self._require_H()
        pts = pixel_pts.astype(np.float32).reshape(1, -1, 2)
        out = cv2.perspectiveTransform(pts, self._H)
        return out.reshape(-1, 2)

    def marker_pose_metric(
        self, corners: np.ndarray
    ) -> tuple[np.ndarray, float]:
        self._require_H()
        metric_corners = self.pixel_to_metric(corners[0])
        centroid = metric_corners.mean(axis=0)
        dx = float(metric_corners[1, 0] - metric_corners[0, 0])
        dy = float(metric_corners[1, 1] - metric_corners[0, 1])
        angle_deg = math.degrees(math.atan2(dy, dx))
        return centroid.astype(np.float32), angle_deg

    def metric_to_pixel(self, metric_pts: np.ndarray) -> np.ndarray:
        self._require_H()
        pts = metric_pts.astype(np.float32).reshape(1, -1, 2)
        out = cv2.perspectiveTransform(pts, self._H_inv)
        return out.reshape(-1, 2)

    def warp_to_zenithal(
        self,
        frame: np.ndarray,
        output_size: tuple[int, int] = config.PIP_SIZE,
        metric_bounds: tuple[float, float, float, float] | None = None,
    ) -> np.ndarray:
        self._require_H()
        if metric_bounds is None:
            if self.metric_bounds is None:
                raise RuntimeError("metric_bounds not set; call compute_homography first.")
            x_min, y_min, x_max, y_max = self.metric_bounds
            dx = (x_max - x_min) * 0.20
            dy = (y_max - y_min) * 0.20
            x_min -= dx; x_max += dx
            y_min -= dy; y_max += dy
        else:
            x_min, y_min, x_max, y_max = metric_bounds

        ow, oh = output_size
        metric_w = x_max - x_min
        metric_h = y_max - y_min
        if metric_w <= 0 or metric_h <= 0:
            return np.full((oh, ow, 3), config.COLORS["pip_bg"], dtype=np.uint8)

        scale = min(ow / metric_w, oh / metric_h)
        offset_x = (ow - metric_w * scale) / 2 - x_min * scale
        offset_y = (oh - metric_h * scale) / 2 + y_max * scale

        S = np.array([
            [ scale,      0,  offset_x],
            [     0, -scale,  offset_y],
            [     0,      0,         1],
        ], dtype=np.float64)

        M = S @ self._H
        bg = np.full((oh, ow, 3), config.COLORS["pip_bg"], dtype=np.uint8)
        zenithal = cv2.warpPerspective(frame, M, (ow, oh), dst=bg,
                                       borderMode=cv2.BORDER_TRANSPARENT)
        self._last_pip_scale = scale
        self._last_pip_offset = (offset_x, offset_y)
        return zenithal

    def metric_to_pip_pixel(self, metric_pts: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_last_pip_scale"):
            raise RuntimeError("Call warp_to_zenithal before metric_to_pip_pixel.")
        scale = self._last_pip_scale
        ox, oy = self._last_pip_offset
        pts = metric_pts.reshape(-1, 2).astype(np.float64)
        pip_pts = np.stack([pts[:, 0] * scale + ox, -pts[:, 1] * scale + oy], axis=1)
        return pip_pts.astype(np.int32)

    @property
    def is_ready(self) -> bool:
        return self._H is not None

    def reset(self) -> None:
        self._H = None
        self._H_inv = None
        self.metric_bounds = None
        self.zenithal_background = None
        for attr in ("_last_pip_scale", "_last_pip_offset"):
            if hasattr(self, attr):
                delattr(self, attr)

    def _require_H(self) -> None:
        if self._H is None:
            raise RuntimeError("Homography not computed. Run RegistrationMode.try_register first.")
