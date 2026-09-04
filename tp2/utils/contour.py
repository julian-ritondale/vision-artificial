import cv2


def get_contours(frame, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE):
    contours, hierarchy = cv2.findContours(frame, mode, method)
    return contours


def filter_contours_by_area(contours, min_area, max_area):
    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            filtered_contours.append(cnt)
    return filtered_contours


def compare_contours(contour_to_compare, saved_contours, max_diff):
    for contour in saved_contours:
        if cv2.matchShapes(contour_to_compare, contour, cv2.CONTOURS_MATCH_I1) < max_diff:
            return True
    return False


def get_bounding_rect(contour):
    return cv2.boundingRect(contour)
