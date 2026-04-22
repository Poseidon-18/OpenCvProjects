import cv2
import numpy as np
from PIL import Image

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # HARDCODED yellow range (typical values)
    lowerLimit = np.array([20, 100, 100])
    upperLimit = np.array([30, 255, 255])
    
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
    cv2.imshow('mask', mask)
    
    # Optional: smooth the mask
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    mask_pil = Image.fromarray(mask)
    bbox = mask_pil.getbbox()
    
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        if (x2 - x1) > 20 and (y2 - y1) > 20:  # Filter small detections
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    cv2.imshow('frame', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()