"""
Rule-based fusion of driver-state and road-state signals into one risk score.
Uses CLIP embeddings to semantically anchor each detected state, then combines
via calibrated weights rather than a trained fusion head (no labeled risk-score
data exists yet for supervised training).
"""

import numpy as np
import torch
import clip

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL, CLIP_PREPROCESS = clip.load("ViT-B/32", device=DEVICE)

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


def clip_semantic_anchor(driver_class, road_summary_text):
    """
    Optional CLIP-based semantic check: embeds a text description of the current
    scene and compares against reference risk-phrase embeddings, as a soft
    consistency signal alongside the rule-based scores (not the primary score).
    """
    reference_phrases = [
        "a safe and alert driving situation",
        "a dangerous driving situation with high risk",
    ]
    scene_text = f"driver state: {driver_class}. road: {road_summary_text}"

    with torch.no_grad():
        text_tokens = clip.tokenize([scene_text] + reference_phrases).to(DEVICE)
        text_features = CLIP_MODEL.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    scene_emb = text_features[0]
    safe_emb, danger_emb = text_features[1], text_features[2] if len(text_features) > 2 else text_features[1]
    sim_danger = (scene_emb @ danger_emb).item()
    return float(np.clip((sim_danger + 1) / 2, 0.0, 1.0))  # rescale cosine [-1,1] to [0,1]


def combine_risk(driver_score, road_score, clip_score=None, driver_weight=0.6, road_weight=0.4):
    """
    Final fused risk score. Driver state weighted higher by default since
    driver impairment is generally a stronger predictor of incident risk
    than ambient road density alone — adjust weights after real testing.
    """
    base = driver_weight * driver_score + road_weight * road_score
    if clip_score is not None:
        base = 0.85 * base + 0.15 * clip_score  # CLIP as a minor semantic nudge, not primary
    return float(np.clip(base, 0.0, 1.0))