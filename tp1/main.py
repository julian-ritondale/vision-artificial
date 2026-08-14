import cv2
import mediapipe as mp
import numpy as np
import platform
from pathlib import Path
import subprocess
import time
import urllib.request

video_path = Path(__file__).with_name("alert.mp4")
audio_path = Path(__file__).with_name("alert.wav")
task_path = Path(__file__).with_name("face_landmarker.task")
video_cap = cv2.VideoCapture(str(video_path))
video_playing = False
audio_process = None

if not task_path.exists():
    print(f"Downloading face_landmarker.task to {task_path}...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    urllib.request.urlretrieve(url, task_path)
    print("Download complete.")

use_tasks_api = False
if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
else:
    use_tasks_api = True
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    base_options = python.BaseOptions(model_asset_path=str(task_path))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

LEFT_IRIS = 468
LEFT_IRIS_TOP = 470
LEFT_CORNERS = (33, 133)
LEFT_TOP = (160, 158)
LEFT_BOTTOM = (144, 153)
LEFT_EYE_CONTOUR = [33, 160, 158, 133, 153, 144]

RIGHT_IRIS = 473
RIGHT_IRIS_TOP = 475
RIGHT_CORNERS = (362, 263)
RIGHT_TOP = (386, 385)
RIGHT_BOTTOM = (373, 380)
RIGHT_EYE_CONTOUR = [362, 385, 386, 263, 380, 373]

MIN_EYE_ASPECT_RATIO = 0.15
MIN_HORIZONTAL_RATIO = 0.35
MAX_HORIZONTAL_RATIO = 0.65
MIN_VERTICAL_OFFSET = -0.10
MAX_VERTICAL_OFFSET = 0
AWAY_FRAMES_TO_TRIGGER = 10

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

    landmarks = None
    if not use_tasks_api:
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
    else:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)
        detection_result = detector.detect_for_video(mp_image, timestamp_ms)
        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]

    looking_away = False
    gaze_unstable = False

    if landmarks:
        lm = landmarks

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

        left_ear = left_height / left_width if left_width > 0 else 0
        right_ear = right_height / right_width if right_width > 0 else 0
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

        # --- DIBUJADO ÚNICO DE LANDMARKS ---
        landmark_color = (0, 0, 255) if looking_away or gaze_unstable else (0, 255, 0)

        # Contorno de ojos
        left_pts = np.array([pt(i) for i in LEFT_EYE_CONTOUR], dtype=np.int32)
        right_pts = np.array([pt(i) for i in RIGHT_EYE_CONTOUR], dtype=np.int32)
        cv2.polylines(frame, [left_pts], True, (255, 255, 255), 1)
        cv2.polylines(frame, [right_pts], True, (255, 255, 255), 1)

        # Centro del iris y círculos
        left_iris_p = pt(LEFT_IRIS)
        right_iris_p = pt(RIGHT_IRIS)
        left_r = max(2, int(left_width * 0.15))
        right_r = max(2, int(right_width * 0.15))

        cv2.circle(frame, (int(left_iris_p[0]), int(left_iris_p[1])), 2, (0, 255, 255), -1)
        cv2.circle(frame, (int(left_iris_p[0]), int(left_iris_p[1])), left_r, landmark_color, 1)

        cv2.circle(frame, (int(right_iris_p[0]), int(right_iris_p[1])), 2, (0, 255, 255), -1)
        cv2.circle(frame, (int(right_iris_p[0]), int(right_iris_p[1])), right_r, landmark_color, 1)
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
