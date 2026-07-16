"""
Lightweight core API for free-tier deployment: driver-state + road-state +
fusion only. The full pipeline (including pothole detection, MiDaS depth,
and video analysis) runs locally — see README for setup instructions.
This split exists purely due to free-tier memory limits (512MB), not
because those features are incomplete; they are fully built and tested,
just not part of this deployed service.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import torch
torch.set_num_threads(1)

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "fusion"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "road_state"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "driver_state"))

import shutil
import uuid

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from fusion_model import driver_risk_score, road_risk_score, combine_risk

app = FastAPI(title="VisionFusion API (Core)")

DRIVER_MODEL_PATH = "../driver_state/models/driver_state/yolov8n_dmd-2/weights/best.pt"
ROAD_MODEL_PATH = "../road_state/models/road_state/yolov8n_bdd100k/weights/best.pt"

driver_model = YOLO(DRIVER_MODEL_PATH)
road_model = YOLO(ROAD_MODEL_PATH)


@app.get("/")
def root():
    return {
        "status": "VisionFusion Core API running",
        "note": "This deployed service covers driver-state + road-state + fusion only. "
                "The full pipeline (pothole detection, MiDaS depth, video analysis) "
                "runs locally — see the project README for setup instructions.",
    }


@app.post("/infer")
async def infer(driver_image: UploadFile = File(...), road_image: UploadFile = File(...)):
    tmp_id = uuid.uuid4().hex
    driver_path = f"tmp_driver_{tmp_id}.jpg"
    road_path = f"tmp_road_{tmp_id}.jpg"

    with open(driver_path, "wb") as f:
        shutil.copyfileobj(driver_image.file, f)
    with open(road_path, "wb") as f:
        shutil.copyfileobj(road_image.file, f)

    try:
        driver_results = driver_model.predict(driver_path, verbose=False)[0]
        road_results = road_model.predict(road_path, verbose=False)[0]

        driver_detections = [
            (driver_results.names[int(box.cls)], float(box.conf))
            for box in driver_results.boxes
        ]
        road_detections = [
            (road_results.names[int(box.cls)], float(box.conf), tuple(box.xywhn[0].tolist()))
            for box in road_results.boxes
        ]

        d_score = driver_risk_score(driver_detections)
        r_score = road_risk_score(road_detections, 1, 1)
        fused = combine_risk(d_score, r_score)

        return JSONResponse({
            "driver_detections": driver_detections,
            "road_detections": road_detections,
            "driver_risk": d_score,
            "road_risk": r_score,
            "fused_risk": fused,
        })
    finally:
        os.remove(driver_path)
        os.remove(road_path)