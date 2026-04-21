"""
enroll_trt.py — Auto-enroll faces from the image_db directory structure.
============================================================================

Directory structure expected:
    image_db/
        sharad/
            sharad_1.jpg
            sharad_2.jpg
        aditya/
            aditya_1.jpg

Behaviour:
    1. Scans every subdirectory of image_db/ as a student name.
    2. If the student already has embeddings in the DB → SKIP entirely.
    3. For new students, extracts embeddings from ALL images in their folder,
       averages them, L2-normalizes, and stores the single fused embedding.
    4. Preprocessing matches the SGIE pipeline exactly (symmetric padding).
"""

import os
import sys
import glob
import cv2
import numpy as np
import onnxruntime as ort
from db_utils import init_db, save_embedding, has_embedding

# ---------- CONFIG ---------- #
MODEL_PATH = "models/w600k_mbf.onnx"
IMAGE_DB_DIR = "image_db"
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')


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
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    # 3. Normalization
    img_float = padded_img.astype(np.float32)
    img_float = (img_float - 127.5) / 128.0
    img_float = np.transpose(img_float, (2, 0, 1))  # HWC -> CHW
    img_float = np.expand_dims(img_float, axis=0)    # (1, 3, 112, 112)
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
        print("    [WARN] No face bounding box found by OpenCV! Using full image (Accuracy may suffer).")
        return img

    # Take largest face
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    x, y, w, h = faces[0]

    print(f"    [INFO] Auto-cropped face at Box={w}x{h}")
    return img[y:y+h, x:x+w]


def enroll_all():
    """
    Auto-scan image_db/ and enroll every student directory.
    
    Idempotency guarantee:
        If a student already has embeddings in the DB, they are 
        COMPLETELY SKIPPED — zero changes to their existing entry.
    """
    init_db()

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: ONNX model not found at {MODEL_PATH}")
        sys.exit(1)

    if not os.path.isdir(IMAGE_DB_DIR):
        print(f"ERROR: image_db directory not found at '{IMAGE_DB_DIR}'")
        print(f"  Create it with subdirectories for each student:")
        print(f"    {IMAGE_DB_DIR}/sharad/sharad_1.jpg")
        print(f"    {IMAGE_DB_DIR}/aditya/aditya_1.jpg")
        sys.exit(1)

    # List all student subdirectories
    student_dirs = sorted([
        d for d in os.listdir(IMAGE_DB_DIR)
        if os.path.isdir(os.path.join(IMAGE_DB_DIR, d))
    ])

    if not student_dirs:
        print(f"ERROR: No student folders found in '{IMAGE_DB_DIR}/'")
        sys.exit(1)

    print("=" * 60)
    print("  AUTO-ENROLLMENT from image_db/")
    print("=" * 60)
    print(f"  Found {len(student_dirs)} student folder(s): {student_dirs}")
    print()

    # Load ONNX session once (CPU to save GPU for the pipeline)
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(MODEL_PATH, providers=providers)
    input_name = session.get_inputs()[0].name

    enrolled_count = 0
    skipped_count = 0

    for student_name in student_dirs:
        student_path = os.path.join(IMAGE_DB_DIR, student_name)

        # ---- IDEMPOTENCY CHECK ---- #
        if has_embedding(student_name):
            print(f"  [SKIP] '{student_name}' — already enrolled in DB (no changes made)")
            skipped_count += 1
            continue

        # Collect all valid images in this student's folder
        image_files = []
        for ext in SUPPORTED_EXTENSIONS:
            image_files.extend(glob.glob(os.path.join(student_path, f'*{ext}')))
            image_files.extend(glob.glob(os.path.join(student_path, f'*{ext.upper()}')))
        # Remove duplicates (case-insensitive glob on Windows)
        image_files = sorted(set(image_files))

        if not image_files:
            print(f"  [WARN] '{student_name}' — folder exists but contains no valid images, skipping")
            continue

        print(f"  [ENROLL] '{student_name}' — processing {len(image_files)} image(s)...")

        # Extract embedding from each image, then average
        embeddings = []
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                print(f"    [WARN] Could not read: {img_path}, skipping")
                continue

            # Extract face crop
            face_img = extract_face(img)

            # Preprocess with DeepStream-equivalent symmetric padding
            input_data = preprocess_face(face_img)

            # Run inference
            output = session.run(None, {input_name: input_data})[0][0]

            # L2 normalize individual embedding
            norm = np.linalg.norm(output)
            if norm > 1e-8:
                output = output / norm

            embeddings.append(output)
            print(f"    ✓ {os.path.basename(img_path)}  dim={len(output)}  "
                  f"norm={np.linalg.norm(output):.4f}  first5={output[:5]}")

        if not embeddings:
            print(f"  [WARN] '{student_name}' — no valid embeddings extracted, skipping")
            continue

        # Average all embeddings for this student, then re-normalize
        avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)
        norm = np.linalg.norm(avg_embedding)
        if norm > 1e-8:
            avg_embedding = avg_embedding / norm

        print(f"    → Fused {len(embeddings)} embedding(s): "
              f"dim={len(avg_embedding)}  norm={np.linalg.norm(avg_embedding):.4f}")

        # Save to database
        save_embedding(student_name, avg_embedding)
        enrolled_count += 1
        print(f"    ✓ SUCCESS: '{student_name}' enrolled into database\n")

    # Summary
    print("=" * 60)
    print(f"  ENROLLMENT COMPLETE")
    print(f"    New enrollments : {enrolled_count}")
    print(f"    Skipped (exists): {skipped_count}")
    print(f"    Total in DB     : {enrolled_count + skipped_count}")
    print("=" * 60)


def enroll_single(image_path, name):
    """Legacy single-image enrollment (backward compatible)."""
    init_db()

    if has_embedding(name):
        print(f"[SKIP] '{name}' already enrolled — no changes made.")
        return

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

    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(MODEL_PATH, providers=providers)
    input_name = session.get_inputs()[0].name

    face_img = extract_face(img)
    input_data = preprocess_face(face_img)

    print(f"Processing: {image_path}")
    print(f"Model: {MODEL_PATH}")

    output = session.run(None, {input_name: input_data})[0][0]

    # L2 normalize
    embedding = output / np.linalg.norm(output)

    print(f"  Dimension: {len(embedding)}")
    print(f"  L2 Norm  : {np.linalg.norm(embedding):.6f}")
    print(f"  First 5  : {embedding[:5]}")

    save_embedding(name, embedding)
    print(f"\nSUCCESS: Enrolled '{name}' into database.")


if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs(IMAGE_DB_DIR, exist_ok=True)

    if len(sys.argv) >= 3:
        # Legacy mode: python3 enroll_trt.py <image_path> <name>
        enroll_single(sys.argv[1], sys.argv[2])
    else:
        # Auto-enrollment mode: python3 enroll_trt.py
        enroll_all()