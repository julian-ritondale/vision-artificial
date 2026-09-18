# TP3 - Planar Homography Localization System

Real-time localization of ArUco fiducial markers in a 2D metric reference
frame using a homography from camera image coordinates to world millimeters.

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
| R | Register the metric plane from one visible ArUco marker |
| Z | Reset the registration |
| Q | Quit |

## Configuration

The main settings are in `config.py`:

```python
ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
MARKER_SIDE_MM = 100.0
CAMERA_SOURCE = 0
```

`MARKER_SIDE_MM` must match the real printed side length of the marker used
for registration.

## Workflow

1. Print or display an ArUco marker from dictionary `DICT_4X4_50`.
2. Set `MARKER_SIDE_MM` to the real marker side length in millimeters.
3. Put the marker flat on the observed plane.
4. Keep the camera fixed.
5. Press `R` while at least one marker is detected.

The detected marker used during registration defines the world frame:

- marker center: `(0, 0)` mm
- marker right direction: positive `x`
- marker upward direction: positive `y`
- marker size: metric scale

After registration, move one or more ArUco markers on the same plane. The
system shows each marker's position in millimeters and orientation angle.

## Windows

- `Cam`: live camera image with detected marker contours, IDs, coordinates,
  and orientation.
- `W2D`: top-down 2D world view using the frozen background captured during
  registration. It draws the world axes, marker square contours, pose arrows,
  coordinates, and orientation labels.

The W2D background is captured only during registration; it is not recomputed
continuously in the camera loop.

## References

- https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
- https://docs.opencv.org/5.0/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
- https://docs.opencv.org/5.0/main_modules/core_array.html#perspectivetransform
