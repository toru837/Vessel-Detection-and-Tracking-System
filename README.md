# Maritime Vessel Detection, Tracking & Re-Identification

**YOLOv12s · ByteTrack · BoT-SORT · DINOv2 · OpenCV · PyTorch**

An AI-based maritime surveillance system developed as part of a research project at **Malaviya National Institute of Technology Jaipur (MNIT Jaipur)**. The project focuses on detecting and tracking vessels in maritime imagery and video, identifying unusual maritime objects, and exploring visual-based vessel re-identification.

The system combines a custom-trained **YOLOv12s** detector with multi-object tracking and a separate deep-learning-based ReID research pipeline.

> **Research Note:** The public repository contains the shareable implementation for vessel detection, tracking, video processing, and anomaly detection. The complete vessel Re-Identification implementation and other research-specific components are not publicly released due to project restrictions.

---

## Overview

Maritime surveillance often involves large amounts of aerial or drone footage that must be monitored continuously. Small vessels, changing viewpoints, camera motion, occlusion, and temporary disappearance of objects make manual monitoring difficult and make consistent vessel identification challenging.

This project explores a computer-vision-based approach that starts with vessel detection and tracking and then extends the detected vessel information into two separate research directions: **anomaly detection** and **vessel re-identification**.

The main workflow is:

```text
Image / Video
      │
      ▼
 YOLOv12s Detection
      │
      ▼
Multi-Object Tracking
      │
      ▼
Track Management
      │
      ▼
Vessel Crops & Track Information
      │
      ├───────────────┐
      ▼               ▼
Anomaly Detection   Vessel ReID
 Notebook            Notebook
```

The anomaly-detection and ReID components are intentionally maintained as separate notebooks because they address different research problems.

---

## Problem Statement

A maritime surveillance system must first determine **what objects are present**, then determine **which detections belong to the same object over time**. Detection alone does not provide persistent identity, while tracking IDs can change when an object disappears and later re-enters the scene.

For example, a vessel may be assigned:

```text
Vessel A
   │
   ├── Frame 100 → Track ID 12
   ├── Frame 101 → Track ID 12
   ├── Frame 102 → Track ID 12
   │
   └── Vessel temporarily leaves the frame
              │
              ▼
        Vessel re-enters
              │
              ▼
        Track ID 57
```

The tracking system can maintain an identity while the vessel remains trackable, while the ReID research investigates whether two observations with different tracking IDs can actually belong to the same physical vessel.

At the same time, some detected objects, such as **windsurfers and SUP boards**, need to be treated differently from conventional vessels. This forms the separate anomaly-detection component of the project.

---

## Detection

The first stage of the system uses a custom-trained **YOLOv12s** model for maritime object detection.

The detector identifies objects from images and video and provides their bounding boxes, classes, and confidence scores. These detections become the input to the tracking stage.

```text
Input Frame
     │
     ▼
  YOLOv12s
     │
     ├── Bounding Box
     ├── Class
     └── Confidence
```

The model was trained on a custom maritime dataset containing vessels and other maritime objects relevant to the project.

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
from ultralytics import YOLO

model = YOLO("best.pt")

model.predict(
    source="video.mp4",
    save=True
)
```

---

## Multi-Object Tracking

After detection, the next problem is maintaining the identity of objects across consecutive frames.

The project explored **ByteTrack** and **BoT-SORT** for multi-object tracking. These trackers associate detections across frames and assign temporary track IDs.

For example:

```text
Frame 1          Frame 2          Frame 3

Vessel A         Vessel A         Vessel A
ID 12            ID 12            ID 12

Vessel B         Vessel B         Vessel B
ID 15            ID 15            ID 15
```

This converts independent frame-level detections into continuous vessel tracks.

The tracking stage also provides vessel crops and track information that can be used by the separate research notebooks.

---

## Vessel Re-Identification

Tracking and ReID solve different problems.

**Tracking** asks:

> Is this detection the same object as in the previous frames?

**ReID** asks:

> Is this newly observed vessel the same physical vessel that was observed earlier?

Consider:

```text
Vessel A
   │
   ▼
Track ID 12
   │
   ▼
Leaves the frame
   │
   ▼
Re-enters the frame
   │
   ▼
Track ID 57
```

The purpose of ReID is to determine whether:

```text
Track ID 57
     │
     ▼
Same physical vessel?
     │
     ▼
Previous Track ID 12
```

The ReID work was developed in a **separate notebook** and is not part of the publicly released implementation.

---

## DINOv2-Based ReID Research

The ReID research explored **DINOv2** as a visual feature extractor.

A vessel crop is passed through DINOv2 to obtain a visual representation. The resulting embeddings can then be compared using cosine similarity.

```text
Vessel Crop
     │
     ▼
  DINOv2
     │
     ▼
Visual Embedding
     │
     ▼
Cosine Similarity
     │
     ▼
Vessel Matching
```

The DINOv2 ViT-B/14 representation used during the research produces a **768-dimensional embedding**.

Rather than relying on a single frame, the research explored maintaining multiple useful visual representations for each vessel. This allows the system to account for differences in viewpoint, illumination, partial occlusion, and vessel appearance.

Conceptually:

```text
Global Vessel ID
       │
       ├── Representative View 1
       ├── Representative View 2
       ├── Representative View 3
       └── Representative View 4
```

The complete ReID implementation, memory-bank logic, and research-specific matching components are intentionally not included in the public repository because of project restrictions.

---

## Anomaly Detection

Anomaly detection is implemented as a **separate notebook** from the ReID work.

The objective is to identify maritime objects that are considered unusual for the intended surveillance scenario. In particular, the project focuses on **windsurfers and SUP boards** as anomalous objects.

The anomaly notebook uses the object-detection output and assigns a different visualization to anomalous and normal detections.

```text
YOLOv12s Detection
        │
        ▼
Detected Object
        │
        ▼
Anomaly Class?
     /       \
   Yes        No
    │          │
    ▼          ▼
 Red Box    Green Box
```

The anomaly notebook is publicly shareable, while the ReID notebook contains research components that cannot be fully released.

---

## Research Organization

The project is organized around a common detection and tracking foundation followed by two independent research directions.

```text
                  Maritime Image / Video
                           │
                           ▼
                     YOLOv12s
                           │
                           ▼
                Multi-Object Tracking
                ByteTrack / BoT-SORT
                           │
                           ▼
                  Vessel Observations
                           │
                 ┌─────────┴─────────┐
                 │                   │
                 ▼                   ▼
       Anomaly Detection        Vessel ReID
          Notebook               Notebook
                 │                   │
                 ▼                   ▼
       Windsurfer / SUP        DINOv2 Features
       Identification          & Similarity
```

This separation is intentional: **anomaly detection and vessel ReID are different research tasks**, even though both use information produced by the detection and tracking stages.

---

## Public Repository

The repository contains the publicly shareable portions of the project, including the YOLOv12s training and inference pipeline, video processing, vessel detection, multi-object tracking, track management, vessel crop generation, and the separate anomaly-detection notebook.

The complete ReID implementation is not publicly available. This includes the complete DINOv2-based matching pipeline, memory-bank implementation, and other research-specific components developed during the project.

This keeps the public repository useful for understanding the detection and tracking system without exposing restricted research code.

---

## Repository Structure

```text
Maritime-Vessel-Detection/
│
├── detection/
│   ├── training/
│   ├── image_inference/
│   └── video_inference/
│
├── tracking/
│   ├── bytetrack/
│   └── botsort/
│
├── anomaly_detection/
│   └── anomaly_detection.ipynb
│
├── re_identification/
│   └── README.md
│
├── data/
│   └── data.yaml
│
├── weights/
│   └── best.pt
│
├── requirements.txt
│
└── README.md
```

The `re_identification` directory can contain documentation describing the research approach without exposing the restricted implementation.

---

## Technologies

**Computer Vision**

`YOLOv12` · `OpenCV` · `ByteTrack` · `BoT-SORT`

**Deep Learning**

`PyTorch` · `DINOv2` · `Vision Transformers`

**Data & Annotation**

`NumPy` · `Pandas` · `Matplotlib` · `Roboflow`

**Development**

`Python` · `Google Colab` · `CUDA`

---

## Applications

The system provides a foundation for automated maritime surveillance, drone-based vessel monitoring, coastal observation, search-and-rescue support, maritime security, and large-scale analysis of aerial maritime video. The detection and tracking components can also serve as a base for future systems that require persistent vessel identity and more advanced maritime intelligence.

---

## Future Research

Future work can focus on improving small-vessel detection, making long-term tracking more robust, improving ReID under significant viewpoint changes, expanding anomaly analysis with motion and stillness information, maintaining larger vessel representation databases, and eventually deploying the complete system in real-time on GPU or edge hardware.

---
## Team

This project was developed as a **three-member research project at Malaviya National Institute of Technology Jaipur (MNIT Jaipur)**. I was one of the three team members and contributed to the development of the maritime computer-vision pipeline, including **vessel detection, multi-object tracking, anomaly detection, and vessel re-identification research**.

## Research Note

The project was carried out as part of a research initiative focused on **intelligent maritime surveillance and search-and-rescue applications**.

The public repository contains the shareable components of the project, including the **vessel detection and tracking implementation** and a **separate anomaly-detection notebook**. The complete **Vessel Re-Identification (ReID)** implementation and certain research-specific components are not publicly released due to project restrictions.

## Author

**Uttam Rathore**
B.Tech, Electronics & Communication Engineering
**Malaviya National Institute of Technology Jaipur (MNIT Jaipur)**
