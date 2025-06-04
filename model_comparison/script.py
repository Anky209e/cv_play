
import os
import time
import cv2
from ultralytics import YOLO
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

# Setup
INPUT_FOLDER = 'images'
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELS = {
    'yolov5n': YOLO('yolov5n.pt'),
    'yolov5s': YOLO('yolov5s.pt')
}

results_summary = []

for model_name, model in MODELS.items():
    detection_counts = defaultdict(int)
    class_diversity = set()
    total_time = 0

    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(INPUT_FOLDER, filename)
            img = cv2.imread(image_path)

            start = time.time()
            results = model(img)[0]
            end = time.time()

            total_time += (end - start)
            for r in results.boxes.data.tolist():
                cls_id = int(r[5])
                label = model.names[cls_id]
                detection_counts[label] += 1
                class_diversity.add(label)

    results_summary.append({
        'Model': model_name,
        'Average Inference Time': total_time / 10,
        'Total Detections': sum(detection_counts.values()),
        'Unique Classes Detected': len(class_diversity)
    })

# Save results
df = pd.DataFrame(results_summary)
df.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'), index=False)

# Plot
df.plot(x='Model', kind='bar', figsize=(10, 5), title='Model Comparison', legend=True)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'))
