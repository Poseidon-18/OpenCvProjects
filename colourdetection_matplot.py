import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def get_limits(color):
    c = np.uint8([[color]])
    hsvColor = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hue = hsvColor[0][0][0]
    lowerLimit = np.array([max(0, int(hue) - 20), 100, 100], dtype=np.uint8) 
    upperLimit = np.array([min(180, int(hue) + 20), 255, 255], dtype=np.uint8)
    return lowerLimit, upperLimit

# BGR format imp hai
COLORS = {
    "Yellow": ([0, 255, 255],(255, 255, 0)),     
    "Red":    ([0, 0, 255],(255, 0, 0)),     
    "Blue":   ([255, 0, 0],(0, 0, 255)),     
    "Green":  ([0, 255, 0],(0, 255, 0)),    
}

cap = cv2.VideoCapture(0)
plt.ion()
fig, ax = plt.subplots()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for color_name, (bgr, box_color) in COLORS.items():
        lowerLimit, upperLimit = get_limits(bgr)
        mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  

        mask_pil = Image.fromarray(mask)
        bbox = mask_pil.getbbox()

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 3)
            
            cv2.putText(display_frame, color_name, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

    ax.clear()
    ax.imshow(display_frame)
    ax.axis('off')
    plt.pause(0.01)

    if not plt.get_fignums():
        break

cap.release()
plt.ioff()
plt.show()