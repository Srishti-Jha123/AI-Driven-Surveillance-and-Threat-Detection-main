from ultralytics import YOLO
import cv2
import urllib.request
import numpy as np
import time
import winsound

# Load YOLO model
model = YOLO("yolov8n.pt")

# Phone camera URL
url = "http://192.0.0.4:8080/shot.jpg"

detect_count = 0  # for stability
last_alert_time=0 #to control buzzer frequency

while True:
    try:
        # Get image from phone
        img_resp = urllib.request.urlopen(url, timeout=1)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        # Run detection
        results = model(frame)

        weapon_detected = False

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]

                # Print detected objects
                print(f"{label} - {conf:.2f}")

                # Bounding box size filter
                x1, y1, x2, y2 = box.xyxy[0]
                area = (x2 - x1) * (y2 - y1)

                # Detection condition (filtered)
                if label in ["knife", "scissors"] and conf > 0.6 and area > 5000:
                    detect_count += 1
                else:
                    detect_count = 0

                # Stable detection
                if detect_count >= 3:
                    weapon_detected = True

        # Draw results
        annotated = results[0].plot()

        # ALERT display
        if weapon_detected:
            cv2.putText(
                annotated,
                "⚠️ WEAPON DETECTED!",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )
            current_time = time.time()

           if current_time - last_alert_time > 3:  # 3 sec gap
               winsound.Beep(1000, 500)
               last_alert_time = current_time
            

        # Show output
        cv2.imshow("Stage 6 - Weapon Detection", annotated)

    except Exception as e:
        print("Stream issue...")

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
