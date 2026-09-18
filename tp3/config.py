import cv2

ARUCO_DICT_ID: int = cv2.aruco.DICT_4X4_50

MARKER_SIDE_MM: float = 100.0

MIN_REF_MARKERS: int = 1

CAMERA_SOURCE: int = 0

FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720

W2D_SIZE: tuple[int, int] = (600, 600)
W2D_HALF_EXTENT_MM: float = 300.0

PIP_SIZE: tuple[int, int] = (360, 360)

PIP_POSITION: str = "top-right"

PIP_MARGIN: int = 10
PIP_BORDER_THICKNESS: int = 2

AXIS_LENGTH_MM: float = 200.0

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_LARGE: float = 0.65
FONT_SCALE_SMALL: float = 0.50
FONT_THICKNESS: int = 2

COLORS: dict[str, tuple[int, int, int]] = {
    "contour_default":  (0, 200, 255),
    "contour_ref":      (0, 255, 80),
    "contour_loc":      (255, 80, 0),
    "contour_missing":  (60, 60, 200),
    "axis_x":           (0, 0, 220),
    "axis_y":           (0, 180, 0),
    "arrow":            (0, 180, 255),
    "text":             (255, 255, 255),
    "text_bg":          (0, 0, 0),
    "pip_border":       (200, 200, 200),
    "status_ok":        (0, 220, 80),
    "status_warn":      (0, 165, 255),
    "status_error":     (0, 0, 220),
    "pip_bg":           (30, 30, 30),
}

ARROW_LENGTH_PX: int = 50

CORNER_CIRCLE_RADIUS: int = 4

KEY_REGISTER: int = ord("r")
KEY_RESET:    int = ord("z")
KEY_QUIT:     int = ord("q")

METRIC_POSITIONS: dict[int, tuple[float, float]] = {
    0: (  0.0,   0.0),
    1: (500.0,   0.0),
    2: (500.0, 500.0),
    3: (  0.0, 500.0),
}
