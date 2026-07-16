from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # pretrained, we fine-tune from here

    model.train(
        data="../../data/bdd100k_subset/data.yaml",
        epochs=30,
        imgsz=512,       # reduced from 640 to fit 4GB VRAM comfortably
        batch=8,         # conservative for 4GB VRAM
        device=0,        # use GPU
        project="../../models/road_state",
        name="yolov8n_bdd100k",
    )

if __name__ == "__main__":
    main()