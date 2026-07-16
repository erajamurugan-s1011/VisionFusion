"""
Measures end-to-end inference latency for each stage of the VisionFusion
pipeline, averaged over multiple runs, on the actual local hardware
(RTX 3050 Laptop GPU, 4GB VRAM).
"""

import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "road_state"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "driver_state"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "pothole_depth"))

from ultralytics import YOLO
from pothole_severity import pothole_model, estimate_depth_map, compute_severity
from fusion_model import driver_risk_score, road_risk_score, combine_risk
from safety_report import build_full_report

DRIVER_MODEL_PATH = "../driver_state/models/driver_state/yolov8n_dmd-2/weights/best.pt"
ROAD_MODEL_PATH = "../road_state/models/road_state/yolov8n_bdd100k/weights/best.pt"

driver_model = YOLO(DRIVER_MODEL_PATH)
road_model = YOLO(ROAD_MODEL_PATH)

DRIVER_TEST_IMG = "../../data/dmd_full/test/images/gA_1_s1_ir_face_mp4-104_jpg.rf.8582442e2178f15714503b42bbd12c84.jpg"
ROAD_TEST_IMG = "../../data/bdd100k_subset/test/images/0010bf16-a457685b_jpg.rf.7e1130a7cd624346758481193ad89067.jpg"
HAZARD_TEST_IMG = "../../data/pothole_full/test/images/img-105_jpg.rf.3fe9dff3d1631e79ecb480ff403bcb86.jpg"

N_RUNS = 20


def time_stage(fn, *args, n=N_RUNS):
    # warmup
    fn(*args)
    start = time.perf_counter()
    for _ in range(n):
        fn(*args)
    elapsed = time.perf_counter() - start
    return (elapsed / n) * 1000  # ms per run


def run_driver():
    return driver_model.predict(DRIVER_TEST_IMG, verbose=False)[0]


def run_road():
    return road_model.predict(ROAD_TEST_IMG, verbose=False)[0]


def run_pothole():
    results = pothole_model.predict(HAZARD_TEST_IMG, verbose=False)[0]
    import cv2
    frame = cv2.imread(HAZARD_TEST_IMG)
    depth_map = estimate_depth_map(frame)
    detections = []
    for box in results.boxes:
        bbox_xyxy = box.xyxy[0].tolist()
        severity_score, label = compute_severity(depth_map, bbox_xyxy, frame.shape)
        detections.append({"bbox": bbox_xyxy, "severity_label": label})
    return detections


if __name__ == "__main__":
    driver_ms = time_stage(run_driver)
    road_ms = time_stage(run_road)
    pothole_ms = time_stage(run_pothole)  # includes MiDaS depth pass
    total_ms = driver_ms + road_ms + pothole_ms

    print(f"Driver-state inference:  {driver_ms:.2f} ms")
    print(f"Road-state inference:    {road_ms:.2f} ms")
    print(f"Pothole + depth:         {pothole_ms:.2f} ms")
    print(f"Total pipeline latency:  {total_ms:.2f} ms  (~{1000/total_ms:.1f} FPS if run sequentially)")