import os
import sys
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

def compute_hu_moments(contour):
    """
    Computes log-scaled Hu moments for a given contour,
    matching the preprocessing in dataset_generator.py.
    """
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    for i in range(7):
        val = float(hu_moments[i])
        mag = abs(val)
        if mag == 0.0:
            hu_moments[i] = 0.0
        else:
            hu_moments[i] = -1.0 * math.copysign(1.0, val) * math.log10(mag)
    return hu_moments

def run_camera_demo(model_path="model/decision_tree_model.joblib", camera_index=0):
    # Support running from either root or tp2/ folder
    if not os.path.exists(model_path):
        alt_path = os.path.join("tp2", model_path)
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise FileNotFoundError(f"Model not found at {model_path} or {alt_path}. Please train the model first.")

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}.")
        return

    window_name = "Tetris Piece Classifier - Live"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    # Trackbars for real-time adjustments
    def nothing(x):
        pass

    cv2.createTrackbar("Block Size", window_name, 33, 100, nothing)     # Adaptive threshold block size (odd > 1)
    cv2.createTrackbar("C Constant", window_name, 2, 50, nothing)        # Adaptive threshold C
    cv2.createTrackbar("Min Area", window_name, 2000, 50000, nothing)    # Minimum contour area
    cv2.createTrackbar("Max Area", window_name, 150000, 300000, nothing) # Maximum contour area
    cv2.createTrackbar("Invert", window_name, 1, 1, nothing)             # 1 = piece is darker than background

    print("\n--- CONTROLS ---")
    print(" Press 'q' or 'ESC' to exit")
    print(" Adjust trackbars to tune contour detection")
    print("----------------\n")

    colors = {
        "cleveland_z": (0, 0, 255),     # Red
        "orange_ricky": (0, 165, 255),  # Orange
        "hero": (255, 255, 0),          # Cyan / Light Blue
        "smashboy": (0, 255, 255),      # Yellow
        "teewee": (255, 0, 255)         # Purple / Magenta
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Read trackbar values
        block_size = cv2.getTrackbarPos("Block Size", window_name)
        if block_size % 2 == 0:
            block_size += 1
        if block_size < 3:
            block_size = 3

        c_val = cv2.getTrackbarPos("C Constant", window_name)
        min_area = cv2.getTrackbarPos("Min Area", window_name)
        max_area = cv2.getTrackbarPos("Max Area", window_name)
        invert = cv2.getTrackbarPos("Invert", window_name)

        # Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bin_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_val
        )

        if invert == 1:
            bin_img = 255 - bin_img

        # Denoising
        kernel = np.ones((3, 3), np.uint8)
        bin_clean = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(bin_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                try:
                    hu_moments = compute_hu_moments(cnt)
                    sample = np.array([hu_moments], dtype=np.float64)
                    
                    # Predict class
                    pred_class = model.predict(sample)[0]
                    label = int_to_label(int(pred_class))
                    color = colors.get(label, (0, 255, 0))

                    # Bounding rectangle and label drawing
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.drawContours(frame, [cnt], -1, color, 3)
                    
                    # Draw label background
                    label_text = f"{label} ({int(area)})"
                    (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w + 6, y), color, -1)
                    cv2.putText(frame, label_text, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

                except Exception:
                    # Ignore ill-conditioned moments or out-of-range shapes
                    pass

        cv2.imshow(window_name, frame)
        cv2.imshow("Binary Mask (Debug)", bin_clean)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera_demo()
