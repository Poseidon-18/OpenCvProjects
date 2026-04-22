import cv2
import numpy as np
from PIL import Image
# was going to do it individually for each colour but this helps jyada
def get_limits(color):
    c = np.uint8([[color]])
    hsvColor = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hue = hsvColor[0][0][0]
    lowerLimit = np.array([max(0, int(hue) - 20), 50, 50], dtype=np.uint8)
    upperLimit = np.array([min(180, int(hue) + 20), 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit
yellow = [0, 255, 255]
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # basically apne liye colour convert kar rhe hai
    lowerLimit, upperLimit = get_limits(color=yellow)
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
    cv2.imshow('mask', mask)  # trying after not getting what i want
    mask_pil = Image.fromarray(mask)
    bbox = mask_pil.getbbox()
    
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    cv2.imshow('frame', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()