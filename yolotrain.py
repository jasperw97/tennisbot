from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data = "mergemoretraining/data.yaml",
    epochs=30,               # enough for tiny dataset
    imgsz=640,
    batch=8
    )
