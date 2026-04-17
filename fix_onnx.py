from ultralytics import YOLO

model = YOLO("yolov8n-face.pt")

model.export(
    format="onnx",
    opset=11,             # This is the "Magic Number" for Jetson Nano
    dynamic=False,        # Forces a fixed 640x640 size (Nano requirement)
    imgsz=[640, 640],     
    simplify=True         # Clean up unnecessary PC-only layers
)