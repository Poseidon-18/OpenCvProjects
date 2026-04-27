#using mediapipe due to isse with fer, so it has been trained and knows what the pixel is like toh greyscale convert karta hai and works like that.
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

# Downloading the face landmarker model as backu[]
MODEL_PATH = "/Users/tanmaybagri/Downloads/OpenCvprojects/face_landmarker.task"


base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

def get_emotion(landmarks, frame_h, frame_w):
    def pt(idx):
        lm = landmarks[idx]
        return lm.x * frame_w, lm.y * frame_h

    left_corner  = pt(61)
    right_corner = pt(291)
    upper_lip    = pt(13)
    lower_lip    = pt(14)

    corners_avg_y = (left_corner[1] + right_corner[1]) / 2
    lip_center_y  = (upper_lip[1] + lower_lip[1]) / 2
    mouth_width   = abs(right_corner[0] - left_corner[0])
    curve_ratio   = (lip_center_y - corners_avg_y) / (mouth_width + 1e-6)

    if curve_ratio > 0.03:
        return "happy", curve_ratio
    elif curve_ratio < -0.06:
        return "sad", curve_ratio
    else:
        return "neutral", curve_ratio


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converting to MediaPipe image format
    rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results    = detector.detect(mp_image)
    h, w       = frame.shape[:2]

    if results.face_landmarks:
        for face in results.face_landmarks:
            emotion, ratio = get_emotion(face, h, w)

            for idx in [61, 291, 13, 14]:
                x = int(face[idx].x * w)
                y = int(face[idx].y * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

            color = {"happy":   (0, 255, 0),
                     "sad":     (255, 0, 0),
                     "neutral": (200, 200, 200)}[emotion]

            cv2.putText(frame, f"{emotion} ({ratio:.3f})",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, color, 2)

    cv2.imshow("Landmark Emotion", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()