import cv2


def denoise(frame, method=cv2.MORPH_ELLIPSE, radius=5):
    if radius <= 0:
        return frame
    kernel = cv2.getStructuringElement(method, (radius, radius))
    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return closing


def apply_color_convertion(frame, color=cv2.COLOR_BGR2GRAY):
    return cv2.cvtColor(frame, color)


def threshold(frame, slider_max, binary, trackbar_value):
    _, th = cv2.threshold(frame, trackbar_value, slider_max, binary)
    return th


def draw_contours(frame, contours, color, thickness=3):
    # -1 for all contours
    cv2.drawContours(frame, contours, -1, color, thickness)
    return frame
