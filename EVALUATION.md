# VisionFusion — Evaluation Report

## Overview
Three independently-trained YOLOv8n detectors + a rule-based fusion layer, benchmarked on real held-out test data. All numbers below are measured, not estimated.

## Model 1: Driver-State Detection (DMD dataset)
- 30 epochs, imgsz 512, batch 8
- **Overall mAP50: 0.978**

| Class            | mAP50 |
|------------------|-------|
| DangerousDriving | 0.994 |
| Distracted       | 0.952 |
| Drinking         | 0.951 |
| SafeDriving      | 0.990 |
| SleepyDriving    | 0.995 |
| Yawn             | 0.985 |

**Known limitation:** Roboflow's DMD split has no session/subject metadata, so train/val leakage across driving sessions cannot be verified or ruled out.

## Model 2: Road-State Detection (BDD100K subset, 4,000 images)
- 30 epochs, imgsz 512, batch 8
- **Overall mAP50: 0.317**

| Class | mAP50 | Instances (val) |
|-------|-------|------------------|
| Car   | 0.665 | 6,012 |
| Truck | 0.382 | 230 |
| Bus   | 0.292 | 118 |
| Person| 0.390 | 956 |
| Rider | 0.287 | 44 |
| Bike  | 0.186 | 61 |
| Motor | 0.021 | 18 |

**Known limitation:** significant class imbalance in the 4K-image subset (car dominates; bike/motor have very few validation instances) — full-dataset training would likely improve rare-class performance. This is a deliberate scope trade-off for a fresher, free-GPU project, not an oversight.

## Model 3: Pothole Detection (665-image dataset)
- 50 epochs, imgsz 512, batch 8
- **mAP50: 0.779, Precision: 0.867, Recall: 0.658**

## Depth/Severity Module
Uses MiDaS_small for monocular depth. **This produces relative depth only** — no camera calibration exists, so absolute size/distance in real-world units is not measured. Severity score combines depth-contrast (pothole region vs. surrounding road) with bounding-box area, mapped to Minor/Moderate/Severe. This is clearly labeled as an approximation in the dashboard UI.

## End-to-End Inference Latency (RTX 3050 Laptop GPU, 4GB VRAM)

| Stage             | Latency (ms) |
|-------------------|--------------|
| Driver-state      | 14.78 |
| Road-state        | 15.60 |
| Pothole + depth   | 34.79 |
| **Total (sequential)** | **65.16 (~15.3 FPS)** |

## Fusion Approach
Rule-based, not trained — no labeled risk-score dataset exists to supervise a fusion head. Driver-state and road-state signals are independently produced (no shared backbone, avoiding gradient interference across dissimilar tasks) and combined via calibrated weights (driver 0.6 / road 0.4), with CLIP embeddings used only as an optional semantic-consistency nudge, not the primary scoring mechanism.

## Future Work (explicitly out of scope for this version)
Seatbelt detection, phone-usage detection, emotion/fatigue estimation, road crack classification, traffic sign detection, road surface/weather classification, lane detection, animal detection, traffic light state, forward collision risk — each would require a dedicated dataset and training pipeline, deliberately deferred to keep this project scoped and shippable.