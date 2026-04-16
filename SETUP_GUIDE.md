# Multi-Camera Face Detection & Attendance Setup Guide

This guide details how to set up the Python environment, download and convert the models, and run the DeepStream pipeline on your Jetson Nano or Desktop system.

## 1. Virtual Environment Setup

Since this is running on an NVIDIA Jetson / Desktop with deepstream installed, we need a virtual environment that can access the system-wide DeepStream `pyds` bindings.

Open your terminal and run:

```bash
# Navigate to the project directory
cd e:/jetson_nano

# Create a virtual environment WITH system site-packages 
# so it inherits the pyds (Python DeepStream) bindings installed by JetPack.
python3 -m venv ds_venv --system-site-packages

# Activate the virtual environment
# On Linux/Jetson: source ds_venv/bin/activate
# On Windows: .\ds_venv\Scripts\activate
```

## 2. Install Required Python Packages

Once the environment is active, install the necessary Python dependencies for database management, matrix operations, and ONNX conversion.

```bash
pip install numpy opencv-python onnxruntime ultralytics
```

## 3. Download and Convert Models

The pipeline requires two ONNX models:
1. `yolov8n-face.onnx`
2. `w600kmbf.onnx`

We have provided a script that downloads the YOLOv8 PYTorch weights (`.pt`), exports them to ONNX via the `ultralytics` package, and sets up the embedding model.

Run the converter script:

```bash
python3 convert_models.py
```

> **Note:** If the automated download links fail due to GitHub rate limits, manually download the YOLOv8-face `.pt` file and place it in the `models/` folder, then rerun the script to trigger the ONNX export.

## 4. DeepStream Custom Parser

To allow DeepStream to understand YOLOv8's native bounding box output grid, you need the custom parser library (`libnvds_infercustomparser_yolov8.so`).

Since you are running DeepStream 6.0.1, you can compile the parser from the DeepStream-Yolo community repository or ensure you place the compiled `.so` file in `e:\jetson_nano\`.

## 5. Enroll Faces

Before running the main pipeline, enroll a face so the database has someone to recognize:

```bash
python3 enroll.py path/to/your/face_image.jpg "John Doe"
```
*This handles resizing to 112x112, calculates the 1D embedding, and inserts it into `attendance.db`.*

## 6. Run the Multi-Camera Pipeline

Once the models are strictly in ONNX format inside the `models/` directory, and your database is populated, launch the pipeline. 

Pass your camera feeds as arguments (RTSP streams or USB v4l2 device paths).

```bash
python3 main_dual_cam.py /dev/video0 /dev/video1
```

If it's successful, DeepStream will boot up, TensorRT will optimize `.engine` files out of your `.onnx` models (this takes ~5 minutes the very first time), and a local window will display the combined camera streams with facial recognition enabled.
