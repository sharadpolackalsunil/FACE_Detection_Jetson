"""
enroll_trt.py — Enroll faces with preprocessing that matches the SGIE exactly.
"""

import os
import sys
import cv2
import numpy as np
import onnxruntime as ort
from db_utils import init_db, save_embedding

# ---------- CONFIG ---------- #
MODEL_PATH = "models/w600k_mbf.onnx"

def preprocess_face(img):
    """
    MUST match SGIE config exactly:
      model-color-format=0         -> RGB
      net-scale-factor=0.0078125   -> 1/128
      offsets=127.5;127.5;127.5
      infer-dims=3;112;112
      symmetric-padding=1          -> Explicitly center-padded!
    """
    # 1. Convert BGR to RGB to match DeepStream's live feed
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. Symmetric Padding (Letterbox)
    h, w = img.shape[:2]
    target_size = 112
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h))
    
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    
    # Deepstream fills borders with 0 (Black)
    padded_img = cv2.copyMakeBorder(
        resized, pad_h, target_size - new_h - pad_h, 
        pad_w, target_size - new_w - pad_w, 
        cv2.BORDER_CONSTANT, value=(0,0,0)
    )
    
    # 3. Normalization
    img_float = padded_img.astype(np.float32)
    img_float = (img_float - 127.5) / 128.0
    img_float = np.transpose(img_float, (2, 0, 1))  # HWC -> CHW
    img_float = np.expand_dims(img_float, axis=0)   # (1, 3, 112, 112)
    return np.ascontiguousarray(img_float, dtype=np.float32)

def extract_face(img):
    """
    Use OpenCV's built-in Haar cascade to dynamically crop the face!
    If we don't crop the photo, the body/background shrinks the face to microscopic sizes!
    """
    # Try to load Haar Cascade from standard OpenCV installations
    import cv2.data
    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        print("  [WARN] No face bounding box found by OpenCV! Using full image (Accuracy may suffer).")
        return img
        
    # Take largest face
    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
    x, y, w, h = faces[0]
    
    print(f"  [INFO] Auto-cropped face at Box={w}x{h}")
    return img[y:y+h, x:x+w]

def enroll(image_path, name):
    """Extract embedding and save to database."""
    init_db()

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: ONNX model not found at {MODEL_PATH}")
        return

    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not read image: {image_path}")
        return

    # Use CPU provider (saves GPU memory for the pipeline)
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(MODEL_PATH, providers=providers)
    input_name = session.get_inputs()[0].name

    # Extract exactly the facial crop to match live YOLO behavior
    face_img = extract_face(img)

    # Preprocess with DeepStream-equivalent symmetric padding
    input_data = preprocess_face(face_img)

    print(f"Processing: {image_path}")
    print(f"Model: {MODEL_PATH}")

    # Run inference
    output = session.run(None, {input_name: input_data})[0][0]

    # L2 normalize
    embedding = output / np.linalg.norm(output)

    print(f"  Dimension: {len(embedding)}")
    print(f"  L2 Norm  : {np.linalg.norm(embedding):.6f}")
    print(f"  First 5  : {embedding[:5]}")

    # Save to database
    save_embedding(name, embedding)
    print(f"\nSUCCESS: Enrolled '{name}' into database.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 enroll_trt.py <path_to_face_image> <person_name>")
        print("Example: python3 enroll_trt.py aditya.jpg Aditya")
        sys.exit(1)

    os.makedirs('models', exist_ok=True)
    enroll(sys.argv[1], sys.argv[2])