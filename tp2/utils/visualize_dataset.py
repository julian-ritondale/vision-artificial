import os
import cv2
import math
import glob
import joblib
import numpy as np
import warnings

warnings.filterwarnings("ignore")

try:
    from utils.label_converters import label_to_int, int_to_label
except ModuleNotFoundError:
    from label_converters import label_to_int, int_to_label

BLOCK_SIZE = 45
C_CONSTANT = 10
LABELS = ["cleveland_z", "hero", "orange_ricky", "smashboy", "teewee"]
COLORS = {
    "cleveland_z":  (0,   0,   255),
    "orange_ricky": (0,   165, 255),
    "hero":         (255, 255, 0),
    "smashboy":     (0,   255, 255),
    "teewee":       (255, 0,   255),
}


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


def process_image(filepath):
    image = cv2.imread(filepath)
    if image is None:
        return None, None, None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, BLOCK_SIZE, C_CONSTANT
    )
    bin_img = 255 - bin_img
    kernel = np.ones((3, 3), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, bin_img, None, None

    contour = max(contours, key=cv2.contourArea)
    hu = compute_hu_moments(contour, bin_img.shape)
    return image, bin_img, contour, hu


def build_display(image, bin_img, contour, true_label, pred_label, index, total, filepath):
    correct = pred_label == true_label
    pred_color = COLORS.get(pred_label, (128, 128, 128)) if pred_label else (128, 128, 128)
    border_color = (0, 200, 0) if correct else (0, 0, 220)

    vis = image.copy()
    if contour is not None:
        cv2.drawContours(vis, [contour], -1, pred_color, 3)

    bin_bgr = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)

    def resize_to_h(img, h):
        ih, iw = img.shape[:2]
        return cv2.resize(img, (int(iw * (h / ih)), h))

    vis_r = resize_to_h(vis, 400)
    bin_r = resize_to_h(bin_bgr, 400)

    divider = np.ones((400, 10, 3), dtype=np.uint8) * 50
    combined = np.hstack([vis_r, divider, bin_r])

    info = np.ones((90, combined.shape[1], 3), dtype=np.uint8) * 25
    status = "CORRECT" if correct else "WRONG"
    status_color = (0, 220, 0) if correct else (0, 0, 220)
    pred_text = pred_label.upper() if pred_label else "NO CONTOUR FOUND"

    cv2.putText(info, f"Predicted: {pred_text}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, pred_color, 2, cv2.LINE_AA)
    cv2.putText(info, f"Ground truth: {true_label.upper()}", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(info, status, (combined.shape[1] - 160, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2, cv2.LINE_AA)

    nav = np.ones((40, combined.shape[1], 3), dtype=np.uint8) * 15
    nav_text = f"  [{index}/{total}]  {os.path.basename(filepath)}    ENTER/SPACE = next   B = back   ESC/Q = quit"
    cv2.putText(nav, nav_text, (8, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1, cv2.LINE_AA)

    border = np.ones((4, combined.shape[1], 3), dtype=np.uint8)
    border[:] = border_color

    return np.vstack([nav, border, combined, info])


def run_visualization(
    model_path="models/decision_tree_model.joblib",
    dataset_root="./tetris",
):
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
        raise FileNotFoundError(f"Model not found. Looked in: {possible_paths}")

    print(f"Loading model from {resolved_path}...")
    model = joblib.load(resolved_path)

    all_files = []
    for label in LABELS:
        for filepath in sorted(glob.glob(os.path.join(dataset_root, label, "*"))):
            all_files.append((filepath, label))

    if not all_files:
        print(f"No images found under {dataset_root}")
        return

    print(f"Found {len(all_files)} images.")
    print("ENTER / SPACE = next   B = back   ESC / Q = quit\n")

    window = "Dataset Visualizer"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    i = 0
    while 0 <= i < len(all_files):
        filepath, true_label = all_files[i]

        image, bin_img, contour, hu = process_image(filepath)
        if image is None:
            print(f"  Could not read {filepath}, skipping.")
            i += 1
            continue

        pred_label = None
        if hu is not None:
            sample = np.array([hu], dtype=np.float64)
            pred_class = model.predict(sample)[0]
            pred_label = int_to_label(int(pred_class))

        correct = pred_label == true_label
        print(f"[{i+1}/{len(all_files)}] {os.path.basename(filepath):<30} "
              f"GT: {true_label:<15} PRED: {str(pred_label):<15} {'OK' if correct else 'WRONG'}")

        frame = build_display(image, bin_img, contour, true_label, pred_label,
                              i + 1, len(all_files), filepath)
        cv2.imshow(window, frame)
        cv2.resizeWindow(window, min(frame.shape[1], 1400), min(frame.shape[0], 700))

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (13, 32):      # Enter or Space -> next
                i += 1
                break
            elif key == ord("b"):    # B -> back
                i = max(0, i - 1)
                break
            elif key in (27, ord("q")):  # ESC or Q -> quit
                cv2.destroyAllWindows()
                return

    print("\nDone.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_visualization()