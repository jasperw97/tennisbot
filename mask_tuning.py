import numpy as np
import matplotlib.pyplot as plt
import cv2
from functions import get_mask_tune, contour_custom_draw, contour_contrast_filter

cap = cv2.VideoCapture("tennis.MPG")

if not cap.isOpened():
    print("Error, can't open video file")
else:
    print("Video opened successfully")

framenum = 0
frames = []

while True:
    success, frame = cap.read()
    frames.append(frame)
    if len(frames) > 2:
        frames = frames[1:]
    if len(frames) == 2:
        mask = get_mask_tune(frames[0], frames[1])
        # copy2 = contour_detect_and_draw(mask, frame)
        cv2.imshow('Mask', mask)
        framenum += 1
        
        if cv2.waitKey(20) == ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()