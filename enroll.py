import os
import sys
import cv2
import numpy as np
import onnxruntime as ort
from db_utils import init_db, save_embedding

def preprocess_face(img):
    """
    Standard preprocessing for MobileFaceNet/ArcFace variants.
    Scales to 112x112, converts to RGB, and normalizes to [-1, 1].
    """
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1)) # HWC to CHW
    img = np.expand_dims(img, axis=0) # Add batch dimension
    img = (img - 127.5) / 128.0
    return img.astype(np.float32)

def enroll(image_path, name, model_path="models/w600kmbf.onnx"):
    """
    Extracts embedding from an image and saves it to the SQLite DB.
    Note: For a robust system, an offline face detector (like RetinaFace/YOLO) 
    should be used here first to crop the face before passing it to MobileFaceNet.
    For demonstration, this assumes the image is already closely cropped to the face.
    """
    init_db()
    
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        return
        
    if not os.path.exists(model_path):
        print(f"Error: ONNX model not found at {model_path}. Please adjust path.")
        return

    # Use CPU execution provider for offline enrollment to save Jetson memory
    providers = ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(model_path, providers=providers)
    input_name = ort_session.get_inputs()[0].name
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to read {image_path}")
        return
        
    # Preprocess and extract
    input_tensor = preprocess_face(img)
    print(f"Extracting embedding for {name}...")
    embedding = ort_session.run(None, {input_name: input_tensor})[0][0]
    
    # L2 Normalization is extremely critical for Cosine Similarity thresholding
    embedding = embedding / np.linalg.norm(embedding)
    
    save_embedding(name, embedding)
    print(f"Successfully enrolled {name} into the database.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 enroll.py <path_to_cropped_image> <user_name>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    user_name = sys.argv[2]
    
    # Defaulting to an expected path for the SGIE model
    # Users will need to ensure they have the ONNX model there
    os.makedirs('models', exist_ok=True)
    enroll(image_path, user_name)
