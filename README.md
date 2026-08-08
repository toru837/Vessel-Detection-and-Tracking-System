# Maritime Vessel Detection, Tracking & Re-Identification

**YOLOv12s · ByteTrack · BoT-SORT · DINOv2 · OpenCV · PyTorch**

An AI-based maritime surveillance system developed as a **three-member research project at Malaviya National Institute of Technology Jaipur (MNIT Jaipur)**. The project focuses on detecting and tracking maritime vessels, identifying unusual objects such as windsurfers and SUP boards, and exploring vessel re-identification using deep visual features.

> **Research Note:** The public repository contains the shareable implementation for vessel detection, tracking, video processing, and anomaly detection. The complete Vessel Re-Identification (ReID) implementation and certain research-specific components are not publicly released due to project restrictions.

---

## Overview

Maritime surveillance involves analyzing large amounts of aerial and drone footage where vessels may be small, visually similar, temporarily occluded, or disappear and re-enter the scene. This project explores a computer-vision approach to automate vessel detection and tracking while extending the system into two separate research areas: **anomaly detection** and **vessel re-identification**.

```text
Image / Video
      |
      v
YOLOv12s Detection
      |
      v
Multi-Object Tracking
ByteTrack / BoT-SORT
      |
      v
Vessel Tracks & Crops
      |
      +-------------------+
      |                   |
      v                   v
Anomaly Detection      Vessel ReID
 Notebook               Notebook
```

The anomaly-detection and ReID work are maintained in separate notebooks because they address different research problems.

---

## Problem Statement

The system addresses the challenges of automatically detecting maritime objects, maintaining their identities across video frames, identifying unusual maritime objects, and exploring whether the same vessel can be recognized after it disappears and re-enters the scene.

A tracking algorithm may assign a new ID when a vessel re-enters the frame:

```text
Vessel A
   |
   v
Track ID 12
   |
   v
Leaves the frame
   |
   v
Re-enters
   |
   v
Track ID 57
```

Tracking handles short-term identity across consecutive frames, while ReID research investigates whether Track ID 57 belongs to the same physical vessel previously assigned Track ID 12.

---

## Vessel Detection

The first stage uses a custom-trained **YOLOv12s** model to detect maritime objects in images and videos. The model provides bounding boxes, object classes, and confidence scores that are passed to the tracking stage.

### Model Configuration

| Parameter  | Value                   |
| ---------- | ----------------------- |
| Model      | YOLOv12s                |
| Framework  | Ultralytics             |
| Epochs     | 13                      |
| Image Size | 640 × 640               |
| Batch Size | 16                      |
| Dataset    | Custom Maritime Dataset |
| Input      | Images and Videos       |

### Training

```python
from ultralytics import YOLO

model = YOLO("yolo12s.pt")

model.train(
    data="data.yaml",
    epochs=13,
    imgsz=640,
    batch=16
)
```

### Image Inference

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model("image.jpg")
results[0].show()
```

### Video Inference

```python
model.predict(
    source="video.mp4",
    save=True
)
```

---

## Multi-Object Tracking

After detection, the system uses **ByteTrack** and **BoT-SORT** to associate detections across consecutive frames and assign temporary track IDs.

```text
Frame 1       Frame 2       Frame 3

Vessel A      Vessel A      Vessel A
ID 12         ID 12         ID 12

Vessel B      Vessel B      Vessel B
ID 15         ID 15         ID 15
```

The tracking stage produces continuous vessel tracks along with track information and vessel crops for further analysis.

---

## Anomaly Detection

Anomaly detection was developed in a **separate notebook** from the ReID work.

The project focuses on identifying unusual maritime objects, particularly **windsurfers and SUP boards**, and distinguishing them from normal maritime detections.

```text
Detected Object
      |
      v
Anomaly Class?
    /     \
  Yes      No
   |        |
 Red Box  Green Box
```

The anomaly-detection notebook is part of the publicly shareable work.

---

## Vessel Re-Identification

The ReID work was developed in a **separate research notebook**.

While tracking attempts to maintain an object's identity across nearby frames, ReID investigates whether two observations with different tracking IDs represent the **same physical vessel**.

The research explored **DINOv2** for extracting visual features from vessel crops.

```text
Vessel Crop
     |
     v
  DINOv2
     |
     v
Visual Embedding
     |
     v
Cosine Similarity
     |
     v
Vessel Matching
```

The DINOv2 ViT-B/14 representation used during the research produces a **768-dimensional embedding**. The research also explored maintaining multiple visual representations of a vessel to handle changes in viewpoint, illumination, partial occlusion, and appearance.

The complete ReID implementation, memory-bank logic, and research-specific matching components are not publicly released due to project restrictions.

---

## Public Repository

The public repository contains the shareable components of the project, including:

* YOLOv12s training and inference
* Maritime vessel detection
* Image and video processing
* ByteTrack and BoT-SORT tracking
* Track management and vessel-crop generation
* Separate anomaly-detection notebook

The complete vessel ReID implementation and other restricted research components are intentionally excluded.

---

## Technologies

**Computer Vision:**
`YOLOv12` · `OpenCV` · `ByteTrack` · `BoT-SORT`

**Deep Learning:**
`PyTorch` · `DINOv2` · `Vision Transformers`

**Data & Tools:**
`NumPy` · `Pandas` · `Matplotlib` · `Roboflow`

**Development:**
`Python` · `Google Colab` · `CUDA`

---

## Research Context

This project was carried out as part of a research initiative focused on **intelligent maritime surveillance and search-and-rescue applications**.

The project was developed by a **three-member team at MNIT Jaipur**, with contributions across the computer-vision, detection, tracking, anomaly-detection, and vessel ReID research components.

---

## Team

**Three-Member Research Team**
Malaviya National Institute of Technology Jaipur (MNIT Jaipur)

I was one of the three team members and contributed to the development of the maritime computer-vision pipeline, including **vessel detection, multi-object tracking, anomaly detection, and vessel re-identification research**.

---

## Author

**Uttam Rathore**
B.Tech, Electronics & Communication Engineering
**Malaviya National Institute of Technology Jaipur (MNIT Jaipur)**

---

## Research Note

Some research-specific implementation details are intentionally excluded from the public repository due to project restrictions.
