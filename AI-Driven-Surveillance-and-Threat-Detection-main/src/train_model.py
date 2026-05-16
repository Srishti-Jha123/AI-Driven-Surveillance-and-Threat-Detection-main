from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

model.train(
    data="../data.yaml",
    epochs=15,
    imgsz=640
)