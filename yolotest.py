from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train/weights/best.pt")

img = cv2.imread("outputimg/output14.png")

results = model(img)

for r in results:
    for box in r.boxes:
        print(box.xyxy[0])