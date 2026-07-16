"""
Rule-based fusion of driver-state and road-state signals into one risk score,
combined via calibrated weights (no trained fusion head — no labeled
risk-score data exists yet for supervised training).
"""

import numpy as np

# Driver-state risk weights (0 = safe, 1 = max risk)
DRIVER_RISK_WEIGHTS = {
    "SafeDriving": 0.0,
    "Distracted": 0.6,
    "Drinking": 0.7,
    "Yawn": 0.5,
    "SleepyDriving": 0.9,
    "DangerousDriving": 1.0,
}

# Road-state object risk weights (per detected instance, scaled by proximity/size)
ROAD_OBJECT_WEIGHTS = {
    "person": 0.9,
    "rider": 0.8,
    "bike": 0.6,
    "motor": 0.6,
    "car": 0.4,
    "bus": 0.5,
    "truck": 0.5,
}


def driver_risk_score(detections):
    """
    detections: list of (class_name, confidence) from the driver-state YOLO model
    Returns a 0-1 risk scalar.
    """
    if not detections:
        return 0.0
    weighted = [DRIVER_RISK_WEIGHTS.get(cls, 0.0) * conf for cls, conf in detections]
    return float(np.clip(max(weighted), 0.0, 1.0))


def road_risk_score(detections, frame_height, frame_width):
    """
    detections: list of (class_name, confidence, bbox) from the road-state YOLO model
    bbox: (x_center, y_center, width, height) normalized 0-1
    Returns a 0-1 risk scalar, weighted higher for objects low in frame (closer) and large (near).
    """
    if not detections:
        return 0.0

    scores = []
    for cls, conf, bbox in detections:
        x_c, y_c, w, h = bbox
        proximity_weight = y_c  # lower in frame = higher y_c = closer
        size_weight = w * h
        base_weight = ROAD_OBJECT_WEIGHTS.get(cls, 0.3)
        scores.append(base_weight * conf * (0.5 + 0.5 * proximity_weight) * (0.5 + 0.5 * min(size_weight * 4, 1.0)))

    return float(np.clip(max(scores), 0.0, 1.0))


def combine_risk(driver_score, road_score, driver_weight=0.6, road_weight=0.4):
    """
    Final fused risk score. Driver state weighted higher by default since
    driver impairment is generally a stronger predictor of incident risk
    than ambient road density alone — adjust weights after real testing.
    """
    base = driver_weight * driver_score + road_weight * road_score
    return float(np.clip(base, 0.0, 1.0))