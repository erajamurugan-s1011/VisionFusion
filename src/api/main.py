import sys
import os
import gc

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "fusion"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "road_state"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "driver_state"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "pothole_depth"))

import cv2
import base64
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from fusion_model import driver_risk_score, road_risk_score, combine_risk
from safety_report import build_full_report

app = FastAPI(title="VisionFusion API")

DRIVER_MODEL_PATH = "../driver_state/models/driver_state/yolov8n_dmd-2/weights/best.pt"
ROAD_MODEL_PATH = "../road_state/models/road_state/yolov8n_bdd100k/weights/best.pt"


def load_driver_model():
    return YOLO(DRIVER_MODEL_PATH)


def load_road_model():
    return YOLO(ROAD_MODEL_PATH)


def load_pothole_module():
    # imported lazily so MiDaS + its weights only load into memory on demand
    import pothole_severity
    return pothole_severity


def release(*objects):
    for obj in objects:
        del obj
    gc.collect()


@app.get("/")
def root():
    return {"status": "VisionFusion API running"}


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
        driver_model = load_driver_model()
        driver_results = driver_model.predict(driver_path, verbose=False)[0]
        driver_detections = [
            (driver_results.names[int(box.cls)], float(box.conf))
            for box in driver_results.boxes
        ]
        release(driver_model)

        road_model = load_road_model()
        road_results = road_model.predict(road_path, verbose=False)[0]
        road_detections = [
            (road_results.names[int(box.cls)], float(box.conf), tuple(box.xywhn[0].tolist()))
            for box in road_results.boxes
        ]
        release(road_model)

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


@app.post("/pothole_infer")
async def pothole_infer(road_hazard_image: UploadFile = File(...)):
    tmp_id = uuid.uuid4().hex
    hazard_path = f"tmp_hazard_{tmp_id}.jpg"

    with open(hazard_path, "wb") as f:
        shutil.copyfileobj(road_hazard_image.file, f)

    try:
        pothole_severity = load_pothole_module()
        detections = pothole_severity.analyze_frame(hazard_path)
        release(pothole_severity)
        return JSONResponse({"pothole_detections": detections})
    finally:
        os.remove(hazard_path)


@app.post("/video_pothole_infer")
async def video_pothole_infer(road_hazard_video: UploadFile = File(...)):
    tmp_id = uuid.uuid4().hex
    video_path = f"tmp_video_{tmp_id}.mp4"

    with open(video_path, "wb") as f:
        shutil.copyfileobj(road_hazard_video.file, f)

    try:
        pothole_severity = load_pothole_module()
        import video_pothole
        result = video_pothole.analyze_video(video_path)
        release(pothole_severity)
        return JSONResponse(result)
    finally:
        os.remove(video_path)


def draw_boxes_and_encode(image_path, detections, color=(0, 255, 0)):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    for bbox, label in detections:
        if len(bbox) == 4 and max(bbox) <= 1.0:
            x_c, y_c, bw, bh = bbox
            x1, y1 = int((x_c - bw / 2) * w), int((y_c - bh / 2) * h)
            x2, y2 = int((x_c + bw / 2) * w), int((y_c + bh / 2) * h)
        else:
            x1, y1, x2, y2 = [int(v) for v in bbox]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, str(label), (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    _, buffer = cv2.imencode(".jpg", img)
    return base64.b64encode(buffer).decode("utf-8")


@app.post("/full_report")
async def full_report(
    driver_image: UploadFile = File(...),
    road_image: UploadFile = File(...),
    road_hazard_image: UploadFile = File(...),
):
    tmp_id = uuid.uuid4().hex
    driver_path = f"tmp_driver_{tmp_id}.jpg"
    road_path = f"tmp_road_{tmp_id}.jpg"
    hazard_path = f"tmp_hazard_{tmp_id}.jpg"

    for upload, path in [(driver_image, driver_path), (road_image, road_path), (road_hazard_image, hazard_path)]:
        with open(path, "wb") as f:
            shutil.copyfileobj(upload.file, f)

    try:
        driver_model = load_driver_model()
        driver_results = driver_model.predict(driver_path, verbose=False)[0]
        driver_detections = [
            (driver_results.names[int(box.cls)], float(box.conf))
            for box in driver_results.boxes
        ]
        driver_boxes_img = draw_boxes_and_encode(
            driver_path,
            [(box.xyxy[0].tolist(), driver_results.names[int(box.cls)]) for box in driver_results.boxes],
            color=(0, 200, 0),
        )
        release(driver_model)

        road_model = load_road_model()
        road_results = road_model.predict(road_path, verbose=False)[0]
        road_detections = [
            (road_results.names[int(box.cls)], float(box.conf), tuple(box.xywhn[0].tolist()))
            for box in road_results.boxes
        ]
        road_boxes_img = draw_boxes_and_encode(
            road_path,
            [(box.xywhn[0].tolist(), road_results.names[int(box.cls)]) for box in road_results.boxes],
            color=(255, 140, 0),
        )
        release(road_model)

        pothole_severity = load_pothole_module()
        pothole_detections = pothole_severity.analyze_frame(hazard_path)
        hazard_boxes_img = draw_boxes_and_encode(
            hazard_path,
            [(p["bbox"], p["severity_label"]) for p in pothole_detections],
            color=(0, 0, 255),
        )
        release(pothole_severity)

        report = build_full_report(driver_detections, road_detections, pothole_detections)

        return JSONResponse({
            "report": report,
            "driver_detections": driver_detections,
            "road_detections": road_detections,
            "pothole_detections": pothole_detections,
            "driver_image_annotated": driver_boxes_img,
            "road_image_annotated": road_boxes_img,
            "hazard_image_annotated": hazard_boxes_img,
        })
    finally:
        for path in [driver_path, road_path, hazard_path]:
            os.remove(path)