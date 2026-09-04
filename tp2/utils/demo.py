import os
import cv2
import math
import joblib
import numpy as np
import warnings

try:
    from utils.label_converters import int_to_label
except ModuleNotFoundError:
    from label_converters import int_to_label

warnings.filterwarnings("ignore")

BLOCK_SIZE_DEFAULT = 45
C_CONSTANT_DEFAULT = 10


def compute_hu_moments(contour, mask_shape):
    filled = np.zeros(mask_shape, dtype=np.uint8)
    cv2.fillPoly(filled, [contour], 255)
    moments = cv2.moments(filled)
    hu_moments = cv2.HuMoments(moments).flatten()
    for i in range(7):
        val = float(hu_moments[i])
        mag = abs(val)
        if mag == 0.0:
            hu_moments[i] = 0.0
        else:
            hu_moments[i] = -1.0 * math.copysign(1.0, val) * math.log10(mag)
    return hu_moments


def run_camera_demo(model_path="models/decision_tree_model.joblib", camera_index=0):
    possible_paths = [
        model_path,
        os.path.join("tp2", model_path),
        "models/decision_tree_model.joblib",
        "tp2/models/decision_tree_model.joblib",
        "model/decision_tree_model.joblib",
        "tp2/model/decision_tree_model.joblib",
    ]
    resolved_path = None
    for p in possible_paths:
        if os.path.exists(p):
            resolved_path = p
            break

    if not resolved_path:
        raise FileNotFoundError(
            f"Model not found. Looked in: {possible_paths}. Please train the model first."
        )

    print(f"Loading model from {resolved_path}...")
    model = joblib.load(resolved_path)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}.")
        return

    window_name = "Tetris Piece Classifier - Live"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 600)

    def nothing(x):
        pass

    cv2.createTrackbar("Block Size", window_name, BLOCK_SIZE_DEFAULT, 100, nothing)
    cv2.createTrackbar("C Constant", window_name, C_CONSTANT_DEFAULT, 50, nothing)
    cv2.createTrackbar("Min Area", window_name, 2500, 50000, nothing)
    cv2.createTrackbar("Max Area", window_name, 120000, 300000, nothing)
    cv2.createTrackbar("Invert", window_name, 1, 1, nothing)

    print("\n--- CONTROLS ---")
    print(f" Block Size default: {BLOCK_SIZE_DEFAULT} | C Constant default: {C_CONSTANT_DEFAULT}")
    print(" Press 'q' or ESC to exit.")
    print("----------------\n")

    colors = {
        "cleveland_z":  (0,   0,   255),
        "orange_ricky": (0,   165, 255),
        "hero":         (255, 255, 0),
        "smashboy":     (0,   255, 255),
        "teewee":       (255, 0,   255),
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        block_size = cv2.getTrackbarPos("Block Size", window_name)
        if block_size % 2 == 0:
            block_size += 1
        if block_size < 3:
            block_size = 3

        c_val = cv2.getTrackbarPos("C Constant", window_name)
        min_area = cv2.getTrackbarPos("Min Area", window_name)
        max_area = cv2.getTrackbarPos("Max Area", window_name)
        invert = cv2.getTrackbarPos("Invert", window_name)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bin_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, c_val
        )

        if invert == 1:
            bin_img = 255 - bin_img

        kernel = np.ones((3, 3), np.uint8)
        bin_clean = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(bin_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = [
            cnt for cnt in contours if min_area <= cv2.contourArea(cnt) <= max_area
        ]

        for cnt in valid_contours:
            try:
                area = cv2.contourArea(cnt)
                hu_moments = compute_hu_moments(cnt, bin_clean.shape)
                sample = np.array([hu_moments], dtype=np.float64)

                pred_class = model.predict(sample)[0]
                label = int_to_label(int(pred_class))
                color = colors.get(label, (0, 255, 0))

                x, y, w, h = cv2.boundingRect(cnt)
                cv2.drawContours(frame, [cnt], -1, color, 3)

                label_text = f"{label.upper()} (Area: {int(area)})"
                (text_w, text_h), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(
                    frame, (x, y - text_h - 12), (x + text_w + 10, y), color, -1
                )
                cv2.putText(
                    frame, label_text, (x + 5, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA,
                )

            except Exception:
                pass

        cv2.imshow(window_name, frame)
        cv2.imshow("Binary Mask (Debug)", bin_clean)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera_demo()