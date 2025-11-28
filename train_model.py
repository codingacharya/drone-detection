from ultralytics import YOLO

def main():
    # Load pretrained YOLOv8s model (lightweight + good for small dataset)
    model = YOLO("yolov8s.pt")

    # Train the model
    model.train(
        data="data.yaml",        # path to your dataset YAML
        epochs=150,              # enough for augmented dataset
        imgsz=640,               # recommended image size
        batch=8,                 # reduce if memory issues
        pretrained=True,
        workers=2,
        device='cpu',            # CPU training (change to '0' if GPU available)
        name="drone_detector",
        patience=30
    )

    print("✅ Training completed! Weights saved in:")
    print("runs/detect/drone_detector/weights/best.pt")

if __name__ == "__main__":
    main()
