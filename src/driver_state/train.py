from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
    data="../../data/dmd_full/data.yaml",
    epochs=30,
    imgsz=512,
    batch=8,
    device=0,
    workers=0,
    project="../../models/driver_state",
    name="yolov8n_dmd",
)

if __name__ == "__main__":
    main()