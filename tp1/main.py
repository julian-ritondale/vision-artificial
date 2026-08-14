import cv2
import mediapipe as mp
import numpy as np
import platform
from pathlib import Path
import subprocess

video_path = Path(__file__).with_name("alert.mp4")
audio_path = Path(__file__).with_name("alert.wav")
video_cap = cv2.VideoCapture(str(video_path))
video_playing = False
audio_process = None

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
AWAY_FRAMES_TO_TRIGGER = 10
LANDMARK_POINTS = (
    LEFT_IRIS,
    LEFT_IRIS_TOP,
    *LEFT_CORNERS,
    *LEFT_TOP,
    *LEFT_BOTTOM,
    RIGHT_IRIS,
    RIGHT_IRIS_TOP,
    *RIGHT_CORNERS,
    *RIGHT_TOP,
    *RIGHT_BOTTOM,
)

cap = cv2.VideoCapture(0)

cv2.namedWindow("Gaze Tracker", cv2.WINDOW_NORMAL)

away_frame_count = 0


def start_alert_audio():
    global audio_process
    stop_alert_audio()

    system_name = platform.system()
    if system_name == "Windows":
        if audio_path.exists():
            import winsound

            winsound.PlaySound(str(audio_path), winsound.SND_ASYNC | winsound.SND_FILENAME)
        return

    if system_name == "Darwin":
        audio_process = subprocess.Popen(["/usr/bin/afplay", str(audio_path if audio_path.exists() else video_path)])
        return

    if audio_path.exists():
        audio_process = subprocess.Popen(["ffplay", "-nodisp", "-autoexit", str(audio_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_alert_audio():
    global audio_process
    system_name = platform.system()

    if system_name == "Windows":
        try:
            import winsound

            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass
        audio_process = None
        return

    if audio_process is not None and audio_process.poll() is None:
        audio_process.terminate()
        try:
            audio_process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            audio_process.kill()
    audio_process = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    looking_away = False
    gaze_unstable = False

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        def pt(i):
            return np.array([lm[i].x * w, lm[i].y * h])

        for landmark_index in LANDMARK_POINTS:
            cv2.circle(frame, pt(landmark_index).astype(int), 2, (0, 255, 0), -1)

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

            looking_away = not (horizontal_ok and vertical_ok)
        else:
            gaze_unstable = True
    else:
        gaze_unstable = True

    if gaze_unstable:
        away_frame_count += 1
    elif looking_away:
        away_frame_count += 1
    else:
        away_frame_count = 0

    alert_active = away_frame_count >= AWAY_FRAMES_TO_TRIGGER

    if alert_active:
        cv2.putText(frame, "LOOK AT SCREEN", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if not video_playing:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            video_playing = True
            start_alert_audio()

        ret_video, video_frame = video_cap.read()
        if ret_video:
            display_frame = video_frame
        else:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            video_playing = False
            start_alert_audio()
            ret_video, video_frame = video_cap.read()
            if ret_video:
                display_frame = video_frame
            else:
                display_frame = frame
    else:
        video_playing = False
        stop_alert_audio()
        display_frame = frame

    cv2.imshow("Gaze Tracker", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_cap.release()
stop_alert_audio()
cv2.destroyAllWindows()
