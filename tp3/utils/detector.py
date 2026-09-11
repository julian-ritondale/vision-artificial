import cv2
import numpy as np
import config

_HAVE_NEW_API: bool = hasattr(cv2.aruco, "ArucoDetector")


class ArucoDetector:

    def __init__(self, dict_id: int = config.ARUCO_DICT_ID) -> None:
        self._dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        self._params = cv2.aruco.DetectorParameters()
        self._params.adaptiveThreshWinSizeMin = 3
        self._params.adaptiveThreshWinSizeMax = 23
        self._params.adaptiveThreshWinSizeStep = 10
        self._params.minMarkerPerimeterRate = 0.03
        self._params.maxMarkerPerimeterRate = 4.0
        if _HAVE_NEW_API:
            self._detector = cv2.aruco.ArucoDetector(self._dictionary, self._params)

    def detect(
        self, frame: np.ndarray
    ) -> tuple[list[np.ndarray], list[int], list[np.ndarray]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if _HAVE_NEW_API:
            raw_corners, raw_ids, rejected = self._detector.detectMarkers(gray)
        else:
            raw_corners, raw_ids, rejected = cv2.aruco.detectMarkers(
                gray, self._dictionary, parameters=self._params
            )
        if raw_ids is None:
            ids: list[int] = []
            corners: list[np.ndarray] = []
        else:
            ids = raw_ids.flatten().tolist()
            corners = list(raw_corners)
        rejected = list(rejected) if rejected is not None else []
        return corners, ids, rejected

    def refine_corners(
        self,
        frame: np.ndarray,
        corners: list[np.ndarray],
        ids: list[int],
    ) -> list[np.ndarray]:
        if not corners:
            return corners
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        refined: list[np.ndarray] = []
        for c in corners:
            pts = c.reshape(-1, 1, 2).astype(np.float32)
            cv2.cornerSubPix(gray, pts, (5, 5), (-1, -1), criteria)
            refined.append(pts.reshape(1, 4, 2))
        return refined

    def detect_and_refine(
        self, frame: np.ndarray
    ) -> tuple[list[np.ndarray], list[int], list[np.ndarray]]:
        corners, ids, rejected = self.detect(frame)
        corners = self.refine_corners(frame, corners, ids)
        return corners, ids, rejected
