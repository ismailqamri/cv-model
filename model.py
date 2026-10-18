
import cv2
import time
from ultralytics import YOLO

model = YOLO("best.pt")

# Try the default camera, then the next available index
cap = None

for index in (0, 1):
    camera = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)

    if camera.isOpened():
        # Give the camera time to initialize
        time.sleep(1)

        success, frame = camera.read()

        if success:
            cap = camera
            print(f"Webcam opened successfully: camera {index}")
            break

    camera.release()

if cap is None:
    raise RuntimeError(
        "Could not read webcam frames. Check camera permissions "
        "and close other applications using the camera."
    )

print("Press 'q' in the video window to quit.")

try:
    while True:
        success, frame = cap.read()

        if not success:
            print("Camera frame unavailable. Retrying...")
            time.sleep(0.2)
            continue

        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv8 Webcam Object Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
