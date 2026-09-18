import cv2

import config

from localization import LocalizationMode
from registration import RegistrationMode
from utils.camera import CameraCapture
from utils.detector import ArucoDetector
from utils.homography import HomographyManager
from utils.renderer import Renderer


def main() -> None:
    try:
        camera = CameraCapture(config.CAMERA_SOURCE)
    except RuntimeError as exc:
        raise SystemExit(f"[ERROR] {exc}")

    detector = ArucoDetector(config.ARUCO_DICT_ID)
    homography = HomographyManager()
    renderer = Renderer()
    registration_mode = RegistrationMode(
        detector, homography, renderer, config.METRIC_POSITIONS
    )
    localization_mode = LocalizationMode(detector, homography, renderer)

    cam_window = "Cam"
    w2d_window = "W2D"
    cv2.namedWindow(cam_window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(w2d_window, cv2.WINDOW_NORMAL)

    print("Homography Localization  |  R = Register  Z = Reset  Q = Quit")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            if not homography.is_ready:
                cam_display = registration_mode.process_frame(frame)
            else:
                cam_display, w2d_display = localization_mode.process_frame(frame)
                if w2d_display is not None:
                    cv2.imshow(w2d_window, w2d_display)

            cv2.imshow(cam_window, cam_display)
            key = cv2.waitKey(1) & 0xFF

            if key == config.KEY_QUIT:
                break
            elif key == config.KEY_REGISTER and not homography.is_ready:
                if not registration_mode.try_register(frame):
                    print("[WARN] Not enough reference markers visible.")
            elif key == config.KEY_RESET:
                homography.reset()

            if cv2.getWindowProperty(cam_window, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
