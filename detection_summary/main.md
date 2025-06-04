
# Task 1: Detection Summary Engine

## 📦 Setup
```python
!pip install -q ultralytics opencv-python matplotlib pandas
```

## 🔧 Imports and Configuration
```python
import cv2
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from collections import defaultdict

# Paths
VIDEO_PATH = 'input_video.mp4'  # Replace with actual path
OUTPUT_DIR = 'output'
ANNOTATED_DIR = os.path.join(OUTPUT_DIR, 'annotated_frames')
os.makedirs(ANNOTATED_DIR, exist_ok=True)
```

## 🚀 Load Pretrained YOLOv5 Model
```python
model = YOLO('yolov5s.pt')  # Loads YOLOv5s
```

## 🎞️ Process Video and Detect Every 5th Frame
```python
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0
detections = {}
class_counter = defaultdict(int)
diversity_record = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 5 == 0:
        results = model(frame)[0]
        frame_json = []
        detected_classes = set()

        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls_id = r
            label = model.names[int(cls_id)]
            detected_classes.add(label)
            class_counter[label] += 1

            frame_json.append({
                'class': label,
                'bbox': [x1, y1, x2, y2],
                'confidence': float(conf)
            })

        detections[frame_count] = frame_json
        diversity_record[frame_count] = len(detected_classes)

        # Save annotated frame
        annotated_frame = results.plot()
        cv2.imwrite(f"{ANNOTATED_DIR}/frame_{frame_count}.jpg", annotated_frame)

        # Save JSON
        with open(f"{OUTPUT_DIR}/frame_{frame_count}.json", "w") as f:
            json.dump(frame_json, f, indent=2)

    frame_count += 1

cap.release()
```

## 📊 Class Frequency Bar Chart
```python
plt.figure(figsize=(10, 5))
pd.Series(class_counter).sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title("Object Frequency")
plt.xlabel("Object Class")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "visualizations/class_frequency.png"))
plt.show()
```

## 🧠 Frame with Maximum Class Diversity
```python
max_div_frame = max(diversity_record, key=diversity_record.get)
print(f"Frame with maximum class diversity: Frame {max_div_frame} with {diversity_record[max_div_frame]} unique classes")
```
