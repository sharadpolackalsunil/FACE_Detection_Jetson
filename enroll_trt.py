"""
enroll_trt.py — Enroll faces using the SAME TensorRT FP16 engine that the SGIE uses.

This guarantees that the embeddings stored in the database are numerically identical
to the ones produced during live inference, fixing the "Unknown" mismatch problem.

Usage:
    python3 enroll_trt.py <path_to_face_image> <person_name>
    python3 enroll_trt.py aditya.jpg Aditya

Requirements (pre-installed on JetPack):
    - tensorrt
    - pycuda
    - opencv-python (cv2)
"""

import os
import sys
import cv2
import numpy as np

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

from db_utils import init_db, save_embedding

# ---------- CONFIG ---------- #
# This MUST match the model-engine-file in configs/sgie_config.txt
ENGINE_PATH = "/home/blackbox/FACE_Detection_Jetson/models/w600k_mbf.onnx_b1_gpu0_fp16.engine"
# Fallback: try local relative path
ENGINE_PATH_LOCAL = "models/w600k_mbf.onnx_b1_gpu0_fp16.engine"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def find_engine():
    """Find the TensorRT engine file."""
    if os.path.exists(ENGINE_PATH):
        return ENGINE_PATH
    if os.path.exists(ENGINE_PATH_LOCAL):
        return ENGINE_PATH_LOCAL
    return None


def load_engine(engine_path):
    """Deserialise a TensorRT engine from disk."""
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def preprocess_face(img):
    """
    MUST match SGIE preprocessing exactly:
      net-scale-factor=0.0078125   →  1/128
      offsets=127.5;127.5;127.5
      model-color-format=0         →  BGR
      infer-dims=3;112;112

    Formula per pixel: (pixel - 127.5) * 0.0078125  =  (pixel - 127.5) / 128.0
    DeepStream feeds BGR (model-color-format=0), so we keep BGR.
    """
    img = cv2.resize(img, (112, 112))
    # DeepStream SGIE uses BGR (model-color-format=0), so do NOT convert to RGB
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)   # Add batch: (1, 3, 112, 112)
    return np.ascontiguousarray(img, dtype=np.float32)


def enroll(image_path, name):
    """Extract embedding using the TensorRT engine and save to database."""
    init_db()

    engine_path = find_engine()
    if engine_path is None:
        print(f"ERROR: TensorRT engine not found at:")
        print(f"  {ENGINE_PATH}")
        print(f"  {ENGINE_PATH_LOCAL}")
        print(f"Run 'python3 main_dual_cam.py /dev/video0' once to auto-generate it.")
        return

    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not read image: {image_path}")
        return

    print(f"Using engine: {engine_path}")
    print(f"Processing: {image_path}")

    # ---- Preprocess ---- #
    input_data = preprocess_face(img)

    # ---- Load engine & create context ---- #
    engine = load_engine(engine_path)
    context = engine.create_execution_context()

    # ---- Discover input/output bindings ---- #
    input_idx = None
    output_idx = None
    for i in range(engine.num_bindings):
        if engine.binding_is_input(i):
            input_idx = i
        else:
            output_idx = i

    input_shape = engine.get_binding_shape(input_idx)
    output_shape = engine.get_binding_shape(output_idx)
    print(f"Engine input  shape: {input_shape}")
    print(f"Engine output shape: {output_shape}")

    # ---- Allocate host + device buffers ---- #
    h_input = np.ascontiguousarray(input_data.reshape(input_shape), dtype=np.float32)
    h_output = np.empty(output_shape, dtype=np.float32)

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)

    stream = cuda.Stream()

    # ---- Run inference ---- #
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v2(
        bindings=[int(d_input), int(d_output)],
        stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

    # ---- Extract and normalise embedding ---- #
    embedding = h_output.flatten()
    norm = np.linalg.norm(embedding)
    if norm < 1e-8:
        print("ERROR: Engine produced a zero embedding. Check your image.")
        return

    embedding = embedding / norm
    print(f"Embedding dim: {len(embedding)}")
    print(f"Embedding L2 norm (should be ~1.0): {np.linalg.norm(embedding):.6f}")
    print(f"First 5 values: {embedding[:5]}")

    # ---- Save to database ---- #
    save_embedding(name, embedding)
    print(f"\nSUCCESS: Enrolled '{name}' using TensorRT FP16 engine.")
    print(f"The embedding will now match the live SGIE output exactly.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 enroll_trt.py <path_to_face_image> <person_name>")
        print("Example: python3 enroll_trt.py aditya.jpg Aditya")
        sys.exit(1)

    enroll(sys.argv[1], sys.argv[2])
