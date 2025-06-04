# Tech at Play Computer vision assignment

## Assignment Overview

This repository contains solutions for **all  three computer vision tasks** as described below. All code, outputs, and deployment instructions are included for reproducibility.

---

### Task 1: Detection Summary Engine

- **Model:** Pretrained YOLOv5.
- **Input:** 15–20 second `.mp4` video.
- **Processing:** Every 5th frame is analyzed.

![image](cv_play/detection_summary/class_frequency.png)

---

### Task 2: Real-Time Stream Simulation & Event Trigger

- **Simulation:** Video stream using OpenCV.
- **Detection:** Every 3rd frame.
- **Alert:** "Crowd Detected" triggered if 3+ people appear in 5 consecutive frames.
- **Logging:** Alerts saved in `.json` or `.txt`.
- **Visualization:** Timeline plot of alert occurrences.

![image](cv_play/realtime_alerts/output/people_count_plot.png)
---

### Task 3: Model Comparison & Docker Deployment

- **Comparison:** Two object detection models (e.g., YOLOv5n vs YOLOv5s) on the same 10 images.
- **Metrics:** Average inference time, detection count, class diversity.
- **Results:** Tabulated comparison.
- **Deployment:** Dockerfile provided for reproducible inference.


---

## Getting Started

### Prerequisites

- Python 3.8+
- Docker (for Task 3)
- [YOLOv5 requirements](https://github.com/ultralytics/yolov5)
- OpenCV, matplotlib, and other dependencies (see `requirements.txt`)

### Installation

```bash
pip install -r requirements.txt
```

### Running the Task 3 Docker Deployment


1. Build the Docker image:
```bash

    docker build -t object-detection-comparison .

```

2. Run inference on a folder of images:
```bash

    docker run --rm -v ${PWD}\output:/app/output object-detection-comparison

```


---

## Outputs

- All outputs (JSON, charts, logs, annotated frames) are saved in the `outputs/` directory.
- See sample results and visualizations in the `outputs/`

---



## References

- [YOLOv5](https://github.com/ultralytics/yolov5)
- [OpenCV](https://opencv.org/)

---