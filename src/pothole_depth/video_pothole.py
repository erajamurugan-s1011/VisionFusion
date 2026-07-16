"""
Video-level pothole analysis: samples frames from an uploaded video at a
fixed rate, runs the existing per-frame pothole+severity pipeline on each,
and aggregates results with timestamps.

Note: this is frame-independent sampling, not object tracking — the same
physical pothole appearing across multiple sampled frames will be counted
each time it appears, not deduplicated. This is a known simplification,
documented here and in the README rather than hidden.
"""

import cv2
import tempfile
import os
from pothole_severity import pothole_model, estimate_depth_map, compute_severity


def analyze_video(video_path, sample_rate_seconds=1.0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(1, int(fps * sample_rate_seconds))

    frame_results = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp_sec = frame_idx / fps
            results = pothole_model.predict(frame, verbose=False)[0]
            depth_map = estimate_depth_map(frame)

            detections = []
            for box in results.boxes:
                bbox_xyxy = box.xyxy[0].tolist()
                conf = float(box.conf)
                severity_score, label = compute_severity(depth_map, bbox_xyxy, frame.shape)
                detections.append({
                    "bbox": bbox_xyxy,
                    "confidence": conf,
                    "severity_score": severity_score,
                    "severity_label": label,
                })

            if detections:
                frame_results.append({
                    "timestamp_sec": round(timestamp_sec, 2),
                    "detections": detections,
                })

        frame_idx += 1

    cap.release()

    total_potholes = sum(len(fr["detections"]) for fr in frame_results)
    all_severities = [d["severity_label"] for fr in frame_results for d in fr["detections"]]
    severity_rank = {"Minor": 0, "Moderate": 1, "Severe": 2}
    worst_severity = max(all_severities, key=lambda s: severity_rank[s]) if all_severities else "None"

    return {
        "frame_results": frame_results,
        "total_detections": total_potholes,
        "worst_severity": worst_severity,
        "frames_analyzed": len(frame_results),
    }


if __name__ == "__main__":
    import sys
    import json
    result = analyze_video(sys.argv[1])
    print(json.dumps(result, indent=2))