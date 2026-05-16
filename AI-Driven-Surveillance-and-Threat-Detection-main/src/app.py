from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import cv2
import urllib.request
import numpy as np

app = Flask(__name__)

# Load your trained model
# model = YOLO("runs/detect/train/weights/best.pt")
model = YOLO("yolov8n.pt")

# Define cameras
CAMERAS = [
    {"id": 1, "name": "Main Entrance", "url": "http://192.0.0.4:8080/shot.jpg"},
    {"id": 2, "name": "Backyard", "url": "http://192.0.0.4:8080/shot.jpg"}, 
    {"id": 3, "name": "Cam 3", "url": "http://192.0.0.4:8080/shot.jpg"}
]

# Global detection status per camera
latest_detections = {cam["id"]: False for cam in CAMERAS}

def generate_frames(camera_id):
    global latest_detections
    
    # Find the URL for the requested camera setup
    camera_url = next((cam["url"] for cam in CAMERAS if cam["id"] == camera_id), None)
    if not camera_url:
        return

    while True:
        try:
            img_resp = urllib.request.urlopen(camera_url, timeout=1)
            img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            results = model(frame)

            weapon_detected = False

            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = model.names[cls]

                    # Detection logic
                    if conf > 0.6:
                        if "Rifle" in label or "Knife" in label or "Handgun" in label:
                            weapon_detected = True

            # Update status for this specific camera
            latest_detections[camera_id] = weapon_detected

            annotated = results[0].plot()

            _, buffer = cv2.imencode('.jpg', annotated)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            continue

@app.route('/')
def index():
    return render_template('index.html', cameras=CAMERAS)

@app.route('/video/<int:camera_id>')
def video(camera_id):
    return Response(generate_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify({"detections": latest_detections})

if __name__ == "__main__":
    # Explicitly enable threading to handle multiple concurrent streams
    app.run(debug=True, threaded=True)