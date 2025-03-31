import cv2
from ultralytics import YOLO

# Load YOLOv8 Nano and force CPU processing
model = YOLO('yolov8n.pt')
model.to('cpu')

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Lower resolution to reduce CPU load
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

frame_count = 0
last_target = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't grab frame, exiting.")
        break

    frame_count += 1

    # Run detection less often to save time
    if frame_count % 5 == 0:
        results = model(frame, classes=[0])[0]  # Class 0 = person

        # If there is at least one detection, use the first one
        if results and hasattr(results, "boxes") and len(results.boxes.xyxy) > 0:
            box = results.boxes.xyxy[0]
            x1, y1, x2, y2 = [int(coord) for coord in box]
            target_x = (x1 + x2) // 2
            target_y = y1 + int(0.1 * (y2 - y1))  # slight downward offset for forehead
            last_target = (target_x, target_y)
        else:
            last_target = None

    if last_target:
        cv2.drawMarker(frame, last_target, (0, 0, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

    cv2.imshow('YOLO CPU Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
