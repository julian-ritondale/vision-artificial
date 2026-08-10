import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

LEFT_IRIS = 468
LEFT_IRIS_TOP = 470
LEFT_CORNERS = (33, 133)
LEFT_TOP = (160, 158)
LEFT_BOTTOM = (144, 153)

RIGHT_IRIS = 473
RIGHT_IRIS_TOP = 475
RIGHT_CORNERS = (362, 263)
RIGHT_TOP = (386, 385)
RIGHT_BOTTOM = (373, 380)

MIN_EYE_ASPECT_RATIO = 0.15
MIN_HORIZONTAL_RATIO = 0.35
MAX_HORIZONTAL_RATIO = 0.65
MIN_VERTICAL_OFFSET = -0.10
MAX_VERTICAL_OFFSET = 0

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    looking_away = True

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        def pt(i):
            return np.array([lm[i].x * w, lm[i].y * h])

        left_top = (pt(LEFT_TOP[0]) + pt(LEFT_TOP[1])) / 2
        left_bottom = (pt(LEFT_BOTTOM[0]) + pt(LEFT_BOTTOM[1])) / 2
        right_top = (pt(RIGHT_TOP[0]) + pt(RIGHT_TOP[1])) / 2
        right_bottom = (pt(RIGHT_BOTTOM[0]) + pt(RIGHT_BOTTOM[1])) / 2

        left_width = np.linalg.norm(pt(LEFT_CORNERS[1]) - pt(LEFT_CORNERS[0]))
        left_height = np.linalg.norm(left_bottom - left_top)
        right_width = np.linalg.norm(pt(RIGHT_CORNERS[1]) - pt(RIGHT_CORNERS[0]))
        right_height = np.linalg.norm(right_bottom - right_top)

        left_ear = left_height / left_width
        right_ear = right_height / right_width
        avg_ear = (left_ear + right_ear) / 2

        if avg_ear > MIN_EYE_ASPECT_RATIO:
            left_x_ratio = np.linalg.norm(pt(LEFT_IRIS) - pt(LEFT_CORNERS[0])) / left_width
            right_x_ratio = np.linalg.norm(pt(RIGHT_IRIS) - pt(RIGHT_CORNERS[0])) / right_width

            left_center_y = (pt(LEFT_CORNERS[0])[1] + pt(LEFT_CORNERS[1])[1]) / 2
            left_y_offset = (pt(LEFT_IRIS)[1] - left_center_y) / left_width

            right_center_y = (pt(RIGHT_CORNERS[0])[1] + pt(RIGHT_CORNERS[1])[1]) / 2
            right_y_offset = (pt(RIGHT_IRIS)[1] - right_center_y) / right_width

            horizontal_ok = (
                MIN_HORIZONTAL_RATIO < left_x_ratio < MAX_HORIZONTAL_RATIO
                and MIN_HORIZONTAL_RATIO < right_x_ratio < MAX_HORIZONTAL_RATIO
            )
            vertical_ok = (
                MIN_VERTICAL_OFFSET < left_y_offset < MAX_VERTICAL_OFFSET
                and MIN_VERTICAL_OFFSET < right_y_offset < MAX_VERTICAL_OFFSET
            )

            if horizontal_ok and vertical_ok:
                looking_away = False

    if looking_away:
        cv2.putText(frame, "LOOK AT SCREEN", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Gaze Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
