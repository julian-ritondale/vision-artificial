import cv2
import numpy as np
import config


class CameraCapture:

    def __init__(self, source: int | str = config.CAMERA_SOURCE) -> None:
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source!r}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    def read(self) -> tuple[bool, np.ndarray]:
        ok, frame = self._cap.read()
        return ok, frame

    def release(self) -> None:
        if self._cap.isOpened():
            self._cap.release()

    @property
    def frame_size(self) -> tuple[int, int]:
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS)

    def __enter__(self) -> "CameraCapture":
        return self

    def __exit__(self, *_) -> None:
        self.release()
