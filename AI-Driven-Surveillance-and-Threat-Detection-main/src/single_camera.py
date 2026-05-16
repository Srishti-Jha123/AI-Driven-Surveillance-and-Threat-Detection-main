import cv2
import urllib.request
import numpy as np

url = "http://192.0.0.4:8080/shot.jpg"

while True:
    try:
        img_resp = urllib.request.urlopen(url, timeout=1)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if frame is None:
            print("Frame decode failed")
            continue

        cv2.imshow("Camera Feed", frame)

    except Exception as e:
        print("Stream issue... retrying")
        continue

    key = cv2.waitKey(1)
    if key == 27:   # ESC key
        break

cv2.destroyAllWindows()