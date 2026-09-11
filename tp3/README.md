# TP3 – Planar Homography Localization System

Real-time localization of ArUco fiducial markers in a 2-D metric reference
frame using OpenCV homography.

## File Structure

```
tp3/
├── main.py           # Entry point
├── config.py         # All configuration (camera, ArUco, colours, metric positions)
├── camera.py         # CameraCapture
├── detector.py       # ArucoDetector
├── homography.py     # HomographyManager
├── registration.py   # RegistrationMode
├── localization.py   # LocalizationMode
└── renderer.py       # Renderer
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| R   | Register – compute homography from visible reference markers |
| Z   | Reset – discard homography, return to registration mode |
| Q   | Quit |

## Configuration

Everything is set in [`config.py`](config.py). The most common things to change:

```python
ARUCO_DICT_ID  = cv2.aruco.DICT_4X4_50   # ArUco dictionary
MARKER_SIDE_MM = 100.0                    # Physical marker side length (mm)
CAMERA_SOURCE  = 0                        # Webcam index

# Metric positions of reference markers (mm)
METRIC_POSITIONS = {
    0: (  0.0,   0.0),
    1: (500.0,   0.0),
    2: (500.0, 500.0),
    3: (  0.0, 500.0),
}
```

## Workflow

### 1 – Registration

1. Print ArUco markers with IDs matching those in `config.METRIC_POSITIONS`.
2. Place them at the exact mm positions defined there
   (e.g. corners of a 500 × 500 mm square on the floor).
3. Point the camera so that all 4 markers are visible.
4. Press **R** to lock the homography.

A green contour indicates a reference marker is detected.
The status bar shows how many reference markers are found.

### 2 – Localization

Once registered the system switches automatically:

* **Camera frame** – blue square contour on each detected marker; label
  showing `ID`, metric `x`, `y` (mm), and orientation `θ` (degrees).
* **Zenithal PIP** (top-right) – bird's-eye view of the metric plane with
  red X / green Y axes and an orange orientation arrow per marker.

Press **Z** to re-register, **Q** to quit.

## Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` | ArUco detection, perspective transforms |
| `numpy` | Array maths |

## References

* ArUco detection: https://docs.opencv.org/4.5.0/d5/dae/tutorial_aruco_detection.html
* Geometric transforms: https://docs.opencv.org/5.0/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
* perspectiveTransform: https://docs.opencv.org/5.0/main_modules/core_array.html#perspectivetransform

