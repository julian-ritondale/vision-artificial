import cv2
import numpy as np

import config
from utils.detector import ArucoDetector
from utils.homography import HomographyManager
from utils.renderer import Renderer


class LocalizationMode:

    def __init__(
        self,
        detector: ArucoDetector,
        homography: HomographyManager,
        renderer: Renderer,
    ) -> None:
        self._detector = detector
        self._homography = homography
        self._renderer = renderer

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:

        out = frame.copy()
        corners, ids, _ = self._detector.detect_and_refine(frame)

        marker_data: list[dict] = []
        for c, mid in zip(corners, ids):
            try:
                centroid_mm, angle_deg = self._homography.marker_pose_metric(c)
            except RuntimeError:
                continue

            metric_corners = self._homography.pixel_to_metric(c[0])

            marker_data.append({
                "id":          mid,
                "corners":     c,
                "metric_corners": metric_corners,
                "centroid_mm": centroid_mm,
                "angle_deg":   angle_deg,
            })

            self._draw_marker_overlay(out, c, mid, centroid_mm, angle_deg)

        self._renderer.draw_mode_banner(out, "localization")
        self._renderer.draw_registration_status(
            out, detected=len(marker_data), required=0, is_registered=True
        )

        if not marker_data:
            return out, None

        return out, self._build_w2d(marker_data)

    def _draw_marker_overlay(
        self,
        frame: np.ndarray,
        corners: np.ndarray,   
        marker_id: int,
        centroid_mm: np.ndarray,  
        angle_deg: float,
    ) -> None:
    
        self._renderer.draw_marker_contour(
            frame, corners,
            color=config.COLORS["contour_loc"],
            thickness=2,
        )
        self._renderer.draw_marker_label(
            frame, corners, marker_id, centroid_mm, angle_deg
        )

    def _build_w2d(
        self,
        marker_data: list[dict],
    ) -> np.ndarray:

        if self._homography.zenithal_background is None:
            raise RuntimeError("W2D background not captured during registration.")

        w2d = self._homography.zenithal_background.copy()

        origin_metric = np.array([[0.0, 0.0]], dtype=np.float32)
        origin_pip = self._homography.metric_to_pip_pixel(origin_metric)[0]
        origin_px = tuple(origin_pip.tolist())

        self._renderer.draw_axes(
            w2d,
            origin_px=origin_px,
            scale=self._homography._last_pip_scale,
            length_mm=config.AXIS_LENGTH_MM,
        )

        w2d = self._draw_zenithal_markers(w2d, marker_data)

        return w2d

    def _draw_zenithal_markers(
        self,
        pip: np.ndarray,
        markers: list[dict],
    ) -> np.ndarray:

        for m in markers:
            centroid_mm: np.ndarray = m["centroid_mm"]
            metric_corners: np.ndarray = m["metric_corners"]
            angle_deg: float = m["angle_deg"]
            marker_id: int = m["id"]

            pip_corners = self._homography.metric_to_pip_pixel(metric_corners)
            pip_pt = self._homography.metric_to_pip_pixel(
                centroid_mm.reshape(1, 2)
            )[0]
            cx, cy = int(pip_pt[0]), int(pip_pt[1])

            ph, pw = pip.shape[:2]
            if not (0 <= cx < pw and 0 <= cy < ph):
                continue

            pts = pip_corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                pip,
                [pts],
                isClosed=True,
                color=config.COLORS["contour_loc"],
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(pip, (cx, cy), 6, config.COLORS["arrow"], -1)

            self._renderer.draw_pose_arrow(
                pip,
                centroid_px=(cx, cy),
                angle_deg=angle_deg,
                color=config.COLORS["arrow"],
            )

            label = (
                f"#{marker_id} x={centroid_mm[0]:.1f}mm "
                f"y={centroid_mm[1]:.1f}mm t={angle_deg:.1f}deg"
            )
            cv2.putText(
                pip, label,
                (cx + 8, cy - 8),
                config.FONT,
                config.FONT_SCALE_SMALL,
                config.COLORS["text"],
                1,
                cv2.LINE_AA,
            )

        return pip
