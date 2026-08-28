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

def run_camera_demo(model_path="models/decision_tree_model.joblib", camera_index=0):
    # Support running from either root or tp2/ folder, and models/ or model/ folder
    possible_paths = [
        model_path,
        os.path.join("tp2", model_path),
        "models/decision_tree_model.joblib",
        "tp2/models/decision_tree_model.joblib",
        "model/decision_tree_model.joblib",
        "tp2/model/decision_tree_model.joblib"
    ]
    resolved_path = None
    for p in possible_paths:
        if os.path.exists(p):
            resolved_path = p
            break

    if not resolved_path:
        raise FileNotFoundError(f"Model not found. Looked in: {possible_paths}. Please train the model first.")

    print(f"Loading model from {resolved_path}...")
    model = joblib.load(resolved_path)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}.")
        return

    window_name = "Tetris Piece Classifier - Live"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 600)

    # Trackbars for real-time adjustments
    def nothing(x):
        pass

    cv2.createTrackbar("Block Size", window_name, 45, 100, nothing)      # Adaptive threshold block size (odd > 1)
    cv2.createTrackbar("C Constant", window_name, 10, 50, nothing)       # Adaptive threshold C (higher = cleaner white)
    cv2.createTrackbar("Min Area", window_name, 2500, 50000, nothing)    # Minimum contour area (filters noise)
    cv2.createTrackbar("Max Area", window_name, 120000, 300000, nothing) # Maximum contour area
    cv2.createTrackbar("Invert", window_name, 1, 1, nothing)             # 1 = piece is dark on light background
    cv2.createTrackbar("Use ROI Box", window_name, 1, 1, nothing)        # 1 = only detect inside center box
    cv2.createTrackbar("Largest Only", window_name, 1, 1, nothing)       # 1 = only classify the largest shape

    print("\n--- CONTROLS & TIPS ---")
    print(" 1. Colocá la pieza dentro del cuadro verde central.")
    print(" 2. Si hay ruido o detecta el fondo, subí 'C Constant' o 'Min Area'.")
    print(" 3. Si la pieza es oscura sobre papel blanco: 'Invert' = 1.")
    print(" 4. Si la pieza es blanca sobre fondo oscuro: 'Invert' = 0.")
    print(" 5. Presioná 'q' o 'ESC' para salir.")
    print("------------------------\n")

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

        h_frame, w_frame = frame.shape[:2]

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
        use_roi = cv2.getTrackbarPos("Use ROI Box", window_name)
        largest_only = cv2.getTrackbarPos("Largest Only", window_name)

        # Region of Interest (ROI)
        if use_roi == 1:
            box_w, box_h = int(w_frame * 0.6), int(h_frame * 0.65)
            roi_x1 = (w_frame - box_w) // 2
            roi_y1 = (h_frame - box_h) // 2
            roi_x2 = roi_x1 + box_w
            roi_y2 = roi_y1 + box_h
            working_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        else:
            roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, w_frame, h_frame
            working_frame = frame

        # Preprocessing
        gray = cv2.cvtColor(working_frame, cv2.COLOR_BGR2GRAY)
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

        # Filter valid contours by area
        valid_contours = [
            cnt for cnt in contours
            if min_area <= cv2.contourArea(cnt) <= max_area
        ]

        # Draw ROI Guide Box
        if use_roi == 1:
            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
            cv2.putText(
                frame, "Presentar figura aqui", (roi_x1 + 10, roi_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        # Select target contours
        if valid_contours:
            if largest_only == 1:
                target_contours = [max(valid_contours, key=cv2.contourArea)]
            else:
                target_contours = valid_contours

            for cnt in target_contours:
                try:
                    area = cv2.contourArea(cnt)
                    hu_moments = compute_hu_moments(cnt)
                    sample = np.array([hu_moments], dtype=np.float64)
                    
                    # Predict class
                    pred_class = model.predict(sample)[0]
                    label = int_to_label(int(pred_class))
                    color = colors.get(label, (0, 255, 0))

                    # Offset contour coordinates to match the full frame
                    cnt_offset = cnt.copy()
                    cnt_offset[:, :, 0] += roi_x1
                    cnt_offset[:, :, 1] += roi_y1

                    # Bounding rectangle and label drawing
                    x, y, w, h = cv2.boundingRect(cnt_offset)
                    cv2.drawContours(frame, [cnt_offset], -1, color, 3)
                    
                    # Draw label background
                    label_text = f"{label.upper()} (Area: {int(area)})"
                    (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, (x, y - text_h - 12), (x + text_w + 10, y), color, -1)
                    cv2.putText(frame, label_text, (x + 5, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

                except Exception:
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
