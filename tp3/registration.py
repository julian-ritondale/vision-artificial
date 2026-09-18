import cv2
import numpy as np

import config
from utils.detector import ArucoDetector
from utils.homography import HomographyManager
from utils.renderer import Renderer


class RegistrationMode:

    def __init__(
        self,
        detector: ArucoDetector,
        homography: HomographyManager,
        renderer: Renderer,
        metric_positions: dict[int, tuple[float, float]],
    ) -> None:
        self._detector = detector
        self._homography = homography
        self._renderer = renderer
        self._metric_positions = metric_positions

        self._last_corners: list[np.ndarray] = []
        self._last_ids: list[int] = []

    def process_frame(self, frame: np.ndarray) -> np.ndarray:

        out = frame.copy()
        corners, ids, rejected = self._detector.detect_and_refine(frame)

        self._last_corners = corners
        self._last_ids = ids

        for c, mid in zip(corners, ids):
            color = config.COLORS["contour_ref"]
            self._renderer.draw_marker_contour(out, c, color, thickness=2)
            self._renderer.draw_marker_label(out, c, mid)

        detected_ref = len(ids)
        required = config.MIN_REF_MARKERS

        self._renderer.draw_registration_status(
            out,
            detected=detected_ref,
            required=required,
            is_registered=self._homography.is_ready,
        )
        self._renderer.draw_mode_banner(out, "registration")

        return out

    def try_register(self, frame: np.ndarray) -> bool:

        corners, ids, _ = self._detector.detect_and_refine(frame)
        self._last_corners = corners
        self._last_ids = ids

        try:
            pixel_pts, metric_pts, marker_id = self._collect_reference_corners(corners, ids)
        except ValueError as exc:
            print(f"[RegistrationMode] Cannot register: {exc}")
            return False

        try:
            self._homography.compute_homography(pixel_pts, metric_pts)
            self._homography.capture_zenithal_background(frame)
            print(
                f"[RegistrationMode] Homography computed from marker #{marker_id}."
            )
            return True
        except (ValueError, RuntimeError) as exc:
            print(f"[RegistrationMode] compute_homography failed: {exc}")
            return False

    def _collect_reference_corners(
        self,
        corners: list[np.ndarray],
        ids: list[int],
    ) -> tuple[np.ndarray, np.ndarray, int]:

        if not corners:
            raise ValueError(
                f"Need {config.MIN_REF_MARKERS} reference markers, "
                "found 0."
            )

        half_side = config.MARKER_SIDE_MM / 2.0
        metric_corners = np.array(
            [
                [-half_side,  half_side],
                [ half_side,  half_side],
                [ half_side, -half_side],
                [-half_side, -half_side],
            ],
            dtype=np.float32,
        )

        return corners[0][0].astype(np.float32), metric_corners, ids[0]
