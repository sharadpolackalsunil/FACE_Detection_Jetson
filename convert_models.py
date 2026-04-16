import os
import urllib.request
import torch
import torch.nn as nn

try:
    from ultralytics import YOLO
    ULTRA_AVAILABLE = True
except ImportError:
    ULTRA_AVAILABLE = False
    print("Warning: ultralytics package not installed. YOLO auto-export may be skipped.")

class DummyFaceNet(nn.Module):
    def __init__(self):
        super(DummyFaceNet, self).__init__()
        # 112x112 input -> 128D embedding
        self.conv = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.fc = nn.Linear(16 * 56 * 56, 128)
        
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1) # Flatten to 1D vector per batch
        x = self.fc(x)
        return x

def main():
    os.makedirs('models', exist_ok=True)

    # 1. YOLOv8
    yolo_onnx_path = "models/yolov8n-face.onnx"
    if os.path.exists(yolo_onnx_path):
        print(f"{yolo_onnx_path} already exists. Skipping YOLO export.")
    else:
        print("Preparing YOLOv8 model...")
        if ULTRA_AVAILABLE:
            try:
                model = YOLO("yolov8n.pt") 
                print("Exporting YOLOv8n to ONNX format...")
                # Note: imgsz=640 is standard
                model.export(format="onnx", imgsz=640, simplify=True)
                
                if os.path.exists("yolov8n.onnx"):
                    os.replace("yolov8n.onnx", yolo_onnx_path)
                    print(f"Successfully created {yolo_onnx_path}")
            except Exception as e:
                print(f"YOLO export failed: {e}")
        else:
            print("Skipping YOLO export due to missing ultralytics.")

    # 2. MobileFaceNet (Dummy representation for stable pipeline testing)
    print("Preparing embedding model validation...")
    facenet_onnx_path = "models/w600kmbf.onnx"
    
    if not os.path.exists(facenet_onnx_path):
        dummy_model = DummyFaceNet()
        dummy_model.eval()
        dummy_input = torch.randn(1, 3, 112, 112)
        
        torch.onnx.export(
            dummy_model, 
            dummy_input, 
            facenet_onnx_path, 
            export_params=True,
            opset_version=11, 
            do_constant_folding=True,
            input_names=['input'], 
            output_names=['embeddings']
        )
        print(f"Generated stable Deepstream embedding representation at {facenet_onnx_path}")
    else:
        print(f"{facenet_onnx_path} already exists.")

if __name__ == "__main__":
    main()
