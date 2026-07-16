"""
Derived analytics layer: converts existing model outputs (driver-state,
road-state, pothole-severity) into a Driver Safety Index, Road Health Index,
overall risk level, natural-language summary, and recommendations.
No new models or datasets — purely a synthesis layer on top of what's
already trained.
"""

DRIVER_CLASS_PENALTY = {
    "SafeDriving": 0,
    "Distracted": 25,
    "Drinking": 30,
    "Yawn": 15,
    "SleepyDriving": 40,
    "DangerousDriving": 50,
}

ROAD_OBJECT_PENALTY = {
    "person": 8,
    "rider": 6,
    "bike": 4,
    "motor": 4,
    "car": 2,
    "bus": 3,
    "truck": 3,
}

SEVERITY_PENALTY = {
    "Minor": 5,
    "Moderate": 15,
    "Severe": 30,
}


def compute_driver_safety_index(driver_detections):
    """driver_detections: list of (class_name, confidence)"""
    if not driver_detections:
        return 100, "Neutral"

    top_class, top_conf = max(driver_detections, key=lambda d: d[1])
    penalty = DRIVER_CLASS_PENALTY.get(top_class, 0) * top_conf
    index = max(0, round(100 - penalty))
    return index, top_class


def compute_road_health_index(road_detections, pothole_detections):
    """
    road_detections: list of (class_name, confidence, bbox)
    pothole_detections: list of dicts with severity_label
    """
    penalty = 0
    for cls, conf, bbox in road_detections:
        x_c, y_c, w, h = bbox
        proximity_weight = 0.5 + 0.5 * y_c
        penalty += ROAD_OBJECT_PENALTY.get(cls, 1) * conf * proximity_weight

    for pothole in pothole_detections:
        penalty += SEVERITY_PENALTY.get(pothole["severity_label"], 0)

    index = max(0, round(100 - penalty))
    return index


def compute_overall_risk(driver_index, road_index):
    combined = (driver_index + road_index) / 2
    if combined >= 75:
        return "Low"
    elif combined >= 50:
        return "Medium"
    else:
        return "High"


def generate_summary(driver_state, road_detections, pothole_detections):
    parts = []

    if driver_state == "SafeDriving":
        parts.append("The driver appears attentive and is driving safely.")
    else:
        parts.append(f"The driver is showing signs of {driver_state.lower()}.")

    if road_detections:
        counts = {}
        for cls, conf, bbox in road_detections:
            counts[cls] = counts.get(cls, 0) + 1
        road_summary = ", ".join(f"{v} {k}(s)" for k, v in counts.items())
        parts.append(f"The road ahead shows {road_summary}.")
    else:
        parts.append("No significant road objects detected.")

    if pothole_detections:
        severe_count = sum(1 for p in pothole_detections if p["severity_label"] == "Severe")
        if severe_count > 0:
            parts.append(f"{severe_count} severe pothole(s) detected on the road surface.")
        else:
            parts.append(f"{len(pothole_detections)} minor/moderate pothole(s) detected.")

    return " ".join(parts)


def generate_recommendations(driver_state, driver_index, road_index, pothole_detections):
    recs = []

    if driver_state in ("SleepyDriving", "DangerousDriving"):
        recs.append("Consider taking a break — signs of impaired driving detected.")
    elif driver_state in ("Distracted", "Drinking", "Yawn"):
        recs.append("Refocus attention on the road.")

    if road_index < 60:
        recs.append("Increase following distance due to nearby traffic/pedestrians.")

    severe_potholes = [p for p in pothole_detections if p["severity_label"] == "Severe"]
    if severe_potholes:
        recs.append("Reduce speed — severe road damage detected ahead.")
    elif pothole_detections:
        recs.append("Stay alert for minor road surface damage.")

    if not recs:
        recs.append("Conditions look safe. Maintain current driving behavior.")

    return recs


def build_full_report(driver_detections, road_detections, pothole_detections):
    driver_index, driver_state = compute_driver_safety_index(driver_detections)
    road_index = compute_road_health_index(road_detections, pothole_detections)
    overall_risk = compute_overall_risk(driver_index, road_index)
    summary = generate_summary(driver_state, road_detections, pothole_detections)
    recommendations = generate_recommendations(driver_state, driver_index, road_index, pothole_detections)

    return {
        "driver_safety_index": driver_index,
        "road_health_index": road_index,
        "overall_risk": overall_risk,
        "summary": summary,
        "recommendations": recommendations,
    }