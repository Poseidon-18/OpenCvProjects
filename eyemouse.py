import cv2
import mediapipe
import pyautogui
face_mesh_landmarks=mediapipe.solutions.face_mesh.FaceMesh(refine_landmarks=True)
cam= cv2.Videocamture(0)
while True:
    _,image=cam.read()
    rgb_image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    output=face_mesh_landmarks.process(rgb_image)
    landmark_points=output.multi_face_landmarks
    print(landmark_points)
    if landmark_points:
        one_landmark_points=landmark_points[0].landmark
        for i in one_landmark_points:
            x=i.x*window_w
            y=i.y*window_h
            print(landmark_points.x,landmark_points.y)
    cv2.imshow("eye mouse",image)
    key=cv2.waitKey(50)
    if key==27:
        break
cam.release()
cv2.destroyAllWindows()
