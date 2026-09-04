import os
import sys
import cv2
import math
import joblib
import numpy as np
import warnings

try:
    from tp2.utils.label_converters import int_to_label
    from tp2.utils.trackbar import create_trackbar, get_trackbar_value
    from tp2.utils.frame_editor import apply_color_convertion, threshold, denoise, draw_contours
    from tp2.utils.contour import get_contours, filter_contours_by_area, get_bounding_rect
except ModuleNotFoundError:
    try:
        from utils.label_converters import int_to_label
        from utils.trackbar import create_trackbar, get_trackbar_value
        from utils.frame_editor import apply_color_convertion, threshold, denoise, draw_contours
        from utils.contour import get_contours, filter_contours_by_area, get_bounding_rect
    except ModuleNotFoundError:
        from label_converters import int_to_label
        from trackbar import create_trackbar, get_trackbar_value
        from frame_editor import apply_color_convertion, threshold, denoise, draw_contours
        from contour import get_contours, filter_contours_by_area, get_bounding_rect

warnings.filterwarnings("ignore")


def compute_hu_moments(contour):
    """
    Computes log-scaled Hu moments for a given contour,
    matching the preprocessing in dataset_generator.py and Fabri's repo.
    """
    mom = cv2.moments(contour)
    hu_moments = cv2.HuMoments(mom).flatten()
    for i in range(7):
        val = float(hu_moments[i])
        mag = abs(val)
        if mag == 0.0:
            hu_moments[i] = 0.0
        else:
            hu_moments[i] = -1.0 * math.copysign(1.0, val) * math.log10(mag)
    return hu_moments


def run_camera_demo(model_path="models/decision_tree_model.joblib", camera_index=0):
    # Support running from either root or tp2/ folder
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

    window_name = "Window"
    debug_window_name = "Window debug"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(debug_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    cv2.resizeWindow(debug_window_name, 600, 450)

    # Trackbars (replicated from Fabri's thresholder & camera design)
    trackbar_thresh_name = "Threshold"
    create_trackbar(trackbar_thresh_name, window_name, slider_max=255, initial_val=127)

    trackbar_kernel_name = "Kernel denoise"
    create_trackbar(trackbar_kernel_name, window_name, slider_max=15, initial_val=5)

    trackbar_min_area_name = "Min Area"
    create_trackbar(trackbar_min_area_name, window_name, slider_max=30000, initial_val=2500)

    trackbar_max_area_name = "Max Area"
    create_trackbar(trackbar_max_area_name, window_name, slider_max=200000, initial_val=120000)

    trackbar_invert_name = "Invert (Dark Piece)"
    create_trackbar(trackbar_invert_name, window_name, slider_max=1, initial_val=1)

    trackbar_largest_name = "Largest Only"
    create_trackbar(trackbar_largest_name, window_name, slider_max=1, initial_val=0)

    print("\n--- FABRI-STYLE THRESHOLDER CONTROLS ---")
    print(" 1. 'Threshold': Ajusta el umbral global de binarización.")
    print(" 2. 'Kernel denoise': Tamaño del elemento estructurante para apertura/clausura morfológica.")
    print(" 3. 'Invert': 1 si la pieza es oscura en fondo claro (papel blanco), 0 si es clara en fondo oscuro.")
    print(" 4. 'Min Area' / 'Max Area': Filtro de ruido y tamaño de contorno.")
    print(" 5. 'Largest Only': 1 para clasificar solo la figura más grande, 0 para todas las válidas.")
    print(" 6. Tecla 'q' o ESC para salir.")
    print("-----------------------------------------\n")

    colors = {
        "cleveland_z": (0, 0, 255),     # Red
        "orange_ricky": (0, 165, 255),  # Orange
        "hero": (255, 255, 0),          # Cyan / Light Blue
        "smashboy": (0, 255, 255),      # Yellow
        "teewee": (255, 0, 255)         # Magenta
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Color conversion
        gray_frame = apply_color_convertion(frame=frame, color=cv2.COLOR_BGR2GRAY)

        # Read trackbar values
        trackbar_thresh_val = get_trackbar_value(trackbar_thresh_name, window_name)
        trackbar_kernel_val = get_trackbar_value(trackbar_kernel_name, window_name)
        trackbar_min_area_val = get_trackbar_value(trackbar_min_area_name, window_name)
        trackbar_max_area_val = get_trackbar_value(trackbar_max_area_name, window_name)
        trackbar_invert_val = get_trackbar_value(trackbar_invert_name, window_name)
        trackbar_largest_val = get_trackbar_value(trackbar_largest_name, window_name)

        # Thresholding (global threshold with invert support)
        binary_type = cv2.THRESH_BINARY_INV if trackbar_invert_val == 1 else cv2.THRESH_BINARY
        thresh_frame = threshold(
            frame=gray_frame,
            slider_max=255,
            binary=binary_type,
            trackbar_value=trackbar_thresh_val
        )

        # Denoising (morphological open & close with ellipse)
        frame_denoised = denoise(frame=thresh_frame, method=cv2.MORPH_ELLIPSE, radius=trackbar_kernel_val)

        # Contours extraction & area filtering
        contours = get_contours(frame=frame_denoised, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        filtered_contours = filter_contours_by_area(
            contours=contours,
            min_area=trackbar_min_area_val,
            max_area=trackbar_max_area_val
        )

        # Prediction and drawing
        if filtered_contours:
            if trackbar_largest_val == 1:
                target_contours = [max(filtered_contours, key=cv2.contourArea)]
            else:
                target_contours = filtered_contours

            for cont in target_contours:
                try:
                    area = cv2.contourArea(cont)
                    hu_moments = compute_hu_moments(cont)
                    sample = np.array([hu_moments], dtype=np.float64)

                    pred = model.predict(sample)[0]
                    label = int_to_label(int(pred))
                    color = colors.get(label, (0, 255, 0))

                    # Draw contour
                    draw_contours(frame=frame, contours=[cont], color=color, thickness=3)

                    # Bounding rect and text
                    x, y, w, h = get_bounding_rect(cont)
                    label_text = f"{label.upper()} ({int(area)})"
                    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x, max(0, y - text_h - 10)), (x + text_w + 6, max(text_h + 10, y)), color, -1)
                    cv2.putText(frame, label_text, (x + 3, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
                except Exception:
                    pass

        cv2.imshow(window_name, frame)
        cv2.imshow(debug_window_name, frame_denoised)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera_demo()
