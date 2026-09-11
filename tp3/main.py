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

    window_name = "Homography Localization"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("Homography Localization  |  R = Register  Z = Reset  Q = Quit")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            if not homography.is_ready:
                display = registration_mode.process_frame(frame)
            else:
                display = localization_mode.process_frame(frame)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key == config.KEY_QUIT:
                break
            elif key == config.KEY_REGISTER and not homography.is_ready:
                if not registration_mode.try_register(frame):
                    print("[WARN] Not enough reference markers visible.")
            elif key == config.KEY_RESET:
                homography.reset()

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

