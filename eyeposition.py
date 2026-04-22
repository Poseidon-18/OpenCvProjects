import cv2
import mediapipe as mp
import pyautogui
import numpy as np
cam = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_PUPIL = 468
RIGHT_PUPIL = 473
screen_w, screen_h = pyautogui.size()
print("Eyemouse starting Press escape to end")

while True:
    _, image = cam.read()
    image = cv2.flip(image, 1)
    h, w = image.shape[:2]
    
    #
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_image)
    
    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark
        
        
        left_pupil = landmarks[LEFT_PUPIL]
        right_pupil = landmarks[RIGHT_PUPIL]
        
        
        left_x, left_y = int(left_pupil.x * w), int(left_pupil.y * h)
        right_x, right_y = int(right_pupil.x * w), int(right_pupil.y * h)
        
        
        cv2.circle(image, (left_x, left_y), 3, (0, 255, 0), -1)
        cv2.circle(image, (right_x, right_y), 3, (0, 255, 0), -1)
        
        
        eye_x = (left_x + right_x) // 2
        eye_y = (left_y + right_y) // 2
        
        screen_x = np.interp(eye_x, [0, w], [0, screen_w])
        screen_y = np.interp(eye_y, [0, h], [0, screen_h])
        
        
        
        cv2.putText(image, f"Eye: ({eye_x}, {eye_y})", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Eye Mouse", image)
    
    key = cv2.waitKey(50)
    if key == 27:  
        break

cam.release()
cv2.destroyAllWindows()