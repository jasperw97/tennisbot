import cv2
from ultralytics import YOLO
from functions import get_mask
import numpy as np

model = YOLO("runs/detect/train3/weights/best.pt")

cap = cv2.VideoCapture("tennis2.MPG")

if not cap.isOpened():
    print("Error, can't open video file")
else:
    print("Video opened successfully")
frames = []
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    #Modify the background?
    frames.append(frame)
    if len(frames) > 2:
        frames = frames[1:]
    if len(frames) == 2:
        mask = get_mask(frames[0], frames[1])
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
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2), 
                mask_roi = mask[y1:y2, x1:x2]
                if np.any(mask_roi != 0):
                
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.rectangle(frame, (int(x1) - 20, int(y1) - 20), (int(x2) + 20, int(y2) + 20), (0, 0, 255), 1)



        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(10) == ord('q'):
            break

