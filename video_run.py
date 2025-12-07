import numpy as np
import matplotlib.pyplot as plt
import cv2
from functions import get_mask, contour_custom_draw, contour_contrast_filter, contour_draw_rect, nms_processed, nms_with_size_select

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
        mask = get_mask(frames[0], frames[1])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        copy = contour_draw_rect(mask, frame)
        copy2 = nms_processed(mask, frame)
        copy3 = nms_with_size_select(mask, frame)
        cv2.imshow('Mask', mask)
        # cv2.imshow('Rect', copy)
        cv2.imshow('NMS', copy2)
        cv2.imshow('NMS+SizeSelect', copy3)
        # cv2.imwrite(f"outputimg/output{framenum}.png", copy2)
        framenum += 1
        
        if cv2.waitKey(20) == ord('q'):
            break
    
cap.release()
cv2.destroyAllWindows()