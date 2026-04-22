import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_limits(color):
    c = np.uint8([[color]])
    hsvColor = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hue = hsvColor[0][0][0]
    lowerLimit = np.array([max(0, int(hue) - 20), 50, 50], dtype=np.uint8)
    upperLimit = np.array([min(180, int(hue) + 20), 255, 255], dtype=np.uint8)
    return lowerLimit, upperLimit

yellow = [0, 255, 255]
cap = cv2.VideoCapture(0)

plt.ion()  # interactive mode on
fig, (ax1, ax2) = plt.subplots(1, 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowerLimit, upperLimit = get_limits(color=yellow)
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    mask_pil = Image.fromarray(mask)
    bbox = mask_pil.getbbox()

    display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # cv2 mein issues ho rha tha toh matplot use kar rhe
    ax1.clear()
    ax2.clear()
    ax1.imshow(display_frame)
    ax1.set_title('Frame')
    ax2.imshow(mask, cmap='gray')
    ax2.set_title('Mask')

    plt.pause(0.01)  

    if not plt.get_fignums():  
        break

cap.release()
plt.ioff()
plt.show()