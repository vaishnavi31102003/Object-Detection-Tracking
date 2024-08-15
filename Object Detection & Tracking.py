import cv2
import torch
from pathlib import Path

# Path to YOLOv5 model weights
model_weights = Path("C:/Users/Admin/Downloads/upload/upload/OBJ_Task1/yolov5s.pt")

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load COCO dataset class names
coco_names = Path("C:/Users/Admin/Downloads/upload/upload/OBJ_Task1/coco.names").read_text().strip().split("\n")
model.names = coco_names

# Initialize video capture (0 for webcam)
cap = cv2.VideoCapture(0)  # Change to 0 if you want to use webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame)

    # Parse detection results
    for result in results.xyxy[0]:
        # Extract bounding box coordinates
        x1, y1, x2, y2, conf, cls = result
        label = model.names[int(cls)]  # Get class label
        color = (0, 255, 0)  # BGR color for the bounding box (green)

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f'{label}: {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame
    cv2.imshow('YOLOv5 Object Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
