"""
Pothole detection + monocular depth estimation, fused into a per-pothole
severity score. Depth is relative (MiDaS output), not metric/calibrated —
this is explicitly an approximation, documented as such.
"""

import torch
import cv2
import numpy as np
from ultralytics import YOLO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import os as _os
POTHOLE_MODEL_PATH = _os.path.join(_os.path.dirname(__file__), "models", "pothole", "yolov8n_pothole-2", "weights", "best.pt")
pothole_model = YOLO(POTHOLE_MODEL_PATH)

# Load MiDaS small (fits comfortably in 4GB VRAM)
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(DEVICE)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform


def estimate_depth_map(image_bgr):
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(DEVICE)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    return depth_map


def compute_severity(depth_map, bbox_xyxy, frame_shape):
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    h, w = frame_shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    pothole_region = depth_map[y1:y2, x1:x2]
    if pothole_region.size == 0:
        return 0.0, "Unknown"

    # Sample a ring around the bbox as the "surrounding road" reference
    pad = int(0.3 * max(x2 - x1, y2 - y1))
    ry1, ry2 = max(0, y1 - pad), min(h, y2 + pad)
    rx1, rx2 = max(0, x1 - pad), min(w, x2 + pad)
    surrounding_region = depth_map[ry1:ry2, rx1:rx2]

    pothole_mean_depth = float(np.mean(pothole_region))
    surrounding_mean_depth = float(np.mean(surrounding_region))

    # MiDaS outputs inverse depth (higher = closer). A pothole is a
    # depression, so it should read as farther (lower value) than the rim.
    depth_contrast = max(0.0, surrounding_mean_depth - pothole_mean_depth)
    normalized_contrast = min(depth_contrast / (surrounding_mean_depth + 1e-6), 1.0)

    bbox_area_frac = ((x2 - x1) * (y2 - y1)) / (w * h)
    size_factor = min(bbox_area_frac * 20, 1.0)  # scale so typical potholes aren't near-zero

    severity_score = float(np.clip(0.7 * normalized_contrast + 0.3 * size_factor, 0.0, 1.0))

    if severity_score < 0.3:
        label = "Minor"
    elif severity_score < 0.6:
        label = "Moderate"
    else:
        label = "Severe"

    return severity_score, label


def analyze_frame(image_path):
    image_bgr = cv2.imread(image_path)
    results = pothole_model.predict(image_path, verbose=False)[0]
    depth_map = estimate_depth_map(image_bgr)

    detections = []
    for box in results.boxes:
        bbox_xyxy = box.xyxy[0].tolist()
        conf = float(box.conf)
        severity_score, label = compute_severity(depth_map, bbox_xyxy, image_bgr.shape)
        detections.append({
            "bbox": bbox_xyxy,
            "confidence": conf,
            "severity_score": severity_score,
            "severity_label": label,
        })

    return detections


if __name__ == "__main__":
    import sys
    import json
    result = analyze_frame(sys.argv[1])
    print(json.dumps(result, indent=2))