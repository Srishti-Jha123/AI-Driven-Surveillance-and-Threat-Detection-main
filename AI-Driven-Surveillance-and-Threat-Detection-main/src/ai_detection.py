from ultralytics import YOLO
import cv2
import urllib.request
import numpy as np

# Load model (downloads automatically first time)
model = YOLO("yolov8n.pt")

url = "http://192.0.0.4:8080/shot.jpg"

while True:
    try:
        img_resp = urllib.request.urlopen(url, timeout=1)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        # Run detection
        results = model(frame)

        # Draw boxes
        annotated = results[0].plot()

        cv2.imshow("AI Detection", annotated)

    except Exception as e:
        print("Stream issue...")

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()