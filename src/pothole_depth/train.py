from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="../../data/pothole_full/data.yaml",
        epochs=50,          # small dataset (465 train images), more epochs helps
        imgsz=512,
        batch=8,
        device=0,
        workers=0,          # avoids the Windows pin-memory CUDA crash we hit before
        project="../../models/pothole",
        name="yolov8n_pothole",
    )

if __name__ == "__main__":
    main()