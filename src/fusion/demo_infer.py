from ultralytics import YOLO
from fusion_model import driver_risk_score, road_risk_score, combine_risk

DRIVER_MODEL_PATH = "../driver_state/models/driver_state/yolov8n_dmd-2/weights/best.pt"
ROAD_MODEL_PATH = "../road_state/models/road_state/yolov8n_bdd100k/weights/best.pt"

driver_model = YOLO(DRIVER_MODEL_PATH)
road_model = YOLO(ROAD_MODEL_PATH)


def run_demo(driver_image_path, road_image_path):
    driver_results = driver_model.predict(driver_image_path, verbose=False)[0]
    road_results = road_model.predict(road_image_path, verbose=False)[0]

    driver_detections = [
        (driver_results.names[int(box.cls)], float(box.conf))
        for box in driver_results.boxes
    ]

    road_detections = [
        (
            road_results.names[int(box.cls)],
            float(box.conf),
            tuple(box.xywhn[0].tolist()),
        )
        for box in road_results.boxes
    ]

    d_score = driver_risk_score(driver_detections)
    r_score = road_risk_score(road_detections, 1, 1)
    fused = combine_risk(d_score, r_score)

    print("Driver detections:", driver_detections)
    print("Road detections:", road_detections)
    print(f"Driver risk: {d_score:.3f}")
    print(f"Road risk:   {r_score:.3f}")
    print(f"FUSED RISK:  {fused:.3f}")


if __name__ == "__main__":
    import sys
    driver_img = sys.argv[1]
    road_img = sys.argv[2]
    run_demo(driver_img, road_img)