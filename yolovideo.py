import cv2
from ultralytics import YOLO
from functions import get_mask, get_valid_contours, reformat, update_roi
import numpy as np

model = YOLO("runs/detect/train3/weights/best.pt")

cap = cv2.VideoCapture("tsitsipas.mp4")

if not cap.isOpened():
    print("Error, can't open video file")
else:
    print("Video opened successfully")
    
    
    
#State Variables
extended_roi = []
yolo_detected = False
framenum = 0
frames = []
detects = []
velocity = [] #initial state velocity

while True:
    framenum += 1
    ret, frame = cap.read()
    frames.append(frame)
    copy = frame.copy()
    if len(velocity) != 0:
        update_roi(extended_roi, velocity)
    
    #Trying to visualize where the roi is to see how well we are trackin the ball
    if len(extended_roi) != 0:
        cv2.rectangle(copy, (int(extended_roi[0]), int(extended_roi[1])), (int(extended_roi[2]), int(extended_roi[3])), (255, 0, 0), 3)
        
    if len(frames) > 2:
        frames = frames[1:]
    if len(frames) == 2:
        mask = get_mask(frames[0], frames[1])
        cv2.imshow("mask", mask)
    # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        results = model(frame, stream=True, conf=0.3)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                #Checking if this detection is within motion mask
                x1, y1, x2, y2 = int(x1) - 3, int(y1) - 3, int(x2) + 3, int(y2) + 3
                mask_roi = mask[y1:y2, x1:x2]
                if np.any(mask_roi != 0):
                    yolo_detected = True
                    #Everytime yolo detects we update it as first_detect
                    detects = [[x1, y1, x2, y2, framenum]]
                    extended_roi = [int(x1) - 20, int(y1) - 20, int(x2) + 20, int(y2) + 20] #declared the first time we get a first-detect
                    
                    cv2.rectangle(copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.rectangle(copy, (int(x1) - 20, int(y1) - 20), (int(x2) + 20, int(y2) + 20), (0, 0, 255), 1)
            # The next frame when YOLO fails to detect
            if len(r.boxes) == 0 and len(detects) != 0:
                contours = get_valid_contours(mask, extended_roi)
                #Right now stick with this so we get one detection, can think of more robust ways in the future
                if len(contours) == 1:
                    current_detect = contours[0]
                    current_detect.append(framenum)
                    detects.append(current_detect)
                    if len(detects) > 2:
                        detects = detects[1:]
                    velocity = [(detects[1][0] - detects[0][0]) / (detects[1][4] - detects[0][4]), (detects[1][1] - detects[0][1]) / (detects[1][4] - detects[0][4])] # [vx, vy]
                    cv2.rectangle(copy, (int(current_detect[0]), int(current_detect[1])), (int(current_detect[2]), int(current_detect[3])), (0, 255, 0), 2)
                
                #First reset methods
                elif len(contours) == 0:
                    # Resets when there is no detections
                    detects = []
                    velocity = []
                    extended_roi = []
                    
                


        # Display the resulting frame
        cv2.imshow('frame', copy)
        if cv2.waitKey(10) == ord('q'):
            break

