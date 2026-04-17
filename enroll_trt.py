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
      model-color-format=0         -> BGR/RGB Alignment
      net-scale-factor=0.0078125   -> 1/128
      offsets=127.5;127.5;127.5
      infer-dims=3;112;112
    """
    img = cv2.resize(img, (112, 112))
    
    # CRITICAL: Convert BGR to RGB to match DeepStream's live feed
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)   # (1, 3, 112, 112)
    return np.ascontiguousarray(img, dtype=np.float32)

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

    # Preprocess
    input_data = preprocess_face(img)

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