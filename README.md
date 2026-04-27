<<<<<<< HEAD
# 🎯 Real-Time Multi-Camera Face Recognition & Attendance System

> **Platform:** NVIDIA Jetson Nano (4GB) · **Framework:** DeepStream 6.0.1 · **Language:** Python 3.6+ (GStreamer bindings)

A production-grade, edge-deployed facial recognition and automated attendance system that runs entirely on a Jetson Nano with zero cloud dependency. The system uses a hardware-accelerated DeepStream pipeline to simultaneously process multiple camera feeds, detect faces with YOLOv8, extract 512-dimensional embeddings with MobileFaceNet (InsightFace w600k), and match identities against a local SQLite database — all in real time at 15–25 FPS.
=======
# 🎯 Real-Time Dual-Camera Face Recognition & Attendance System

> **Platform:** NVIDIA Jetson Nano (4GB) · **Framework:** DeepStream 6.0.1 · **Language:** Python 3.6+ (GStreamer bindings)

A production-grade, edge-deployed facial recognition and automated attendance system that runs entirely on a Jetson Nano with zero cloud dependency. The system uses a hardware-accelerated DeepStream pipeline to simultaneously process **two camera feeds** (USB, RTSP, HTTP, or HTTPS), detect faces with YOLOv8, extract 512-dimensional embeddings with MobileFaceNet (InsightFace w600k), and match identities against a local SQLite database — all in real time at 15–25 FPS per camera.
>>>>>>> sharad

---

## 📑 Table of Contents

1. [System Architecture](#-system-architecture)
2. [Pipeline Deep Dive](#-pipeline-deep-dive)
3. [AI Models Used](#-ai-models-used)
4. [Key Features](#-key-features)
5. [Project Structure](#-project-structure)
6. [File-by-File Breakdown](#-file-by-file-breakdown)
7. [Configuration Deep Dive](#-configuration-deep-dive)
8. [How Everything Connects](#-how-everything-connects)
9. [Setup & Installation](#-setup--installation)
10. [Usage Guide](#-usage-guide)
11. [Performance Tuning](#-performance-tuning)
12. [Troubleshooting](#-troubleshooting)

---

## 🏗 System Architecture

```
<<<<<<< HEAD
┌─────────────────────────────────────────────────────────────────────────┐
│                         JETSON NANO (Edge Device)                       │
│                                                                         │
│  ┌──────────┐  ┌──────────┐                                            │
│  │ Camera 0 │  │ Camera 1 │   ← USB / RTSP / HTTP                     │
│  │/dev/video0│ │/dev/video1│                                            │
│  └────┬─────┘  └────┬─────┘                                            │
│       │              │                                                  │
│       ▼              ▼                                                  │
│  ┌──────────────────────────┐                                          │
│  │     nvstreammux           │  ← Batches frames from all cameras      │
│  │  (640×480, batch=N)       │                                          │
│  └────────────┬──────────────┘                                          │
│               ▼                                                         │
│  ┌──────────────────────────┐                                          │
│  │   PGIE — YOLOv8n-face    │  ← Primary GPU Inference Engine          │
│  │  (Face Detection)         │     Detects bounding boxes               │
│  │  FP16 · batch=16         │     Custom NMS parser (.so)              │
│  └────────────┬──────────────┘                                          │
│               ▼                                                         │
│  ┌──────────────────────────┐                                          │
│  │   nvtracker (IOU)         │  ← Assigns persistent Object IDs        │
│  │   640×640 tracking grid   │     Enables frame-to-frame identity     │
│  └────────────┬──────────────┘                                          │
│               ▼                                                         │
│  ┌──────────────────────────┐                                          │
│  │  SGIE — MobileFaceNet    │  ← Secondary GPU Inference Engine        │
│  │  (512-D Embedding)        │     Crops each face → 112×112           │
│  │  FP16 · symmetric-pad    │     Outputs normalized embedding         │
│  └────────────┬──────────────┘                                          │
│               ▼                                                         │
│  ┌──────────────────────────┐                                          │
│  │  Python Probe Function    │  ← Main application logic               │
│  │  • Vectorized matching    │     Cosine similarity (matrix multiply)  │
│  │  • Tracker ID cache       │     Attendance logging                   │
│  │  • Cross-camera dedup     │     OSD overlay rendering                │
│  └────────────┬──────────────┘                                          │
│               ▼                                                         │
│  ┌──────────────────────────┐                                          │
│  │  nvmultistreamtiler       │  ← Side-by-side view of all cameras     │
│  │  → nvdsosd → eglglessink  │     Green/Red boxes + Name labels       │
│  └───────────────────────────┘                                          │
│                                                                         │
│  ┌───────────────┐  ┌────────────────┐                                 │
│  │ attendance.db  │  │ attendance.csv │  ← Dual output logging          │
│  │  (SQLite)      │  │  (Spreadsheet) │                                 │
│  └───────────────┘  └────────────────┘                                 │
└─────────────────────────────────────────────────────────────────────────┘
=======
┌──────────────────────────────────────────────────────────────────────────┐
│                          JETSON NANO (Edge Device)                       │
│                                                                          │
│  ┌────────────────┐          ┌────────────────┐                         │
│  │   Camera 0      │          │   Camera 1      │                       │
│  │  /dev/video0    │          │  /dev/video1    │                        │
│  │  USB / HTTP     │          │  RTSP / HTTPS   │                        │
│  └───────┬────────┘          └───────┬────────┘                         │
│          │ v4l2src / uridecodebin    │ v4l2src / uridecodebin           │
│          │                           │                                   │
│          ▼                           ▼                                   │
│     nvvideoconvert              nvvideoconvert                           │
│      (→ NVMM NV12)              (→ NVMM NV12)                           │
│          │                           │                                   │
│          └───────────┬───────────────┘                                   │
│                      ▼                                                   │
│          ┌──────────────────────┐                                       │
│          │    nvstreammux        │  ← Batches frames from BOTH cameras  │
│          │  (640×480, batch=N)   │     N = num cameras (adaptive)       │
│          └──────────┬───────────┘                                       │
│                     ▼                                                    │
│          ┌──────────────────────┐                                       │
│          │  PGIE — YOLOv8n-face │  ← Primary GPU Inference Engine       │
│          │  (Face Detection)     │     FP16 · batch=16                  │
│          │  Custom NMS parser    │     Detects ALL face bounding boxes   │
│          └──────────┬───────────┘                                       │
│                     ▼                                                    │
│          ┌──────────────────────┐                                       │
│          │  nvtracker (IOU)      │  ← Assigns persistent Object IDs     │
│          │  640×640 tracking     │     Per-camera ID isolation            │
│          └──────────┬───────────┘                                       │
│                     ▼                                                    │
│          ┌──────────────────────┐                                       │
│          │  SGIE — MobileFaceNet│  ← Secondary GPU Inference Engine     │
│          │  (512-D Embedding)    │     Crops each face → 112×112        │
│          │  FP16 · symmetric-pad│     Outputs normalized embedding      │
│          └──────────┬───────────┘                                       │
│                     ▼                                                    │
│          ┌──────────────────────────────────────────┐                   │
│          │       Python Probe Function               │                  │
│          │                                           │                  │
│          │  • Vectorized matching (matrix @ vector)  │                  │
│          │  • Tracker ID cache (per-camera keys)     │                  │
│          │  • Cross-camera attendance dedup           │                  │
│          │  • OSD overlay (green/red boxes + names)  │                  │
│          └──────────┬───────────────────────────────┘                   │
│                     ▼                                                    │
│          ┌──────────────────────┐                                       │
│          │ nvmultistreamtiler    │  ← Side-by-side: Cam 0 | Cam 1      │
│          │ (1280×720 grid)       │                                       │
│          └──────────┬───────────┘                                       │
│                     ▼                                                    │
│              nvdsosd → nveglglessink                                     │
│              (Green/Red boxes)   (HDMI Display)                          │
│                                                                          │
│  ┌─────────────────┐    ┌──────────────────┐                            │
│  │  attendance.db   │    │  attendance.csv   │  ← Dual output logging   │
│  │  (SQLite)        │    │  (Name,Date,Time) │     5-min CSV dedup      │
│  └─────────────────┘    └──────────────────┘                            │
└──────────────────────────────────────────────────────────────────────────┘
>>>>>>> sharad
```

---

## 🔬 Pipeline Deep Dive

The GStreamer pipeline is the heart of this system. Every frame travels through the following stages in hardware-accelerated order:

### Stage 1: Source Acquisition
<<<<<<< HEAD
Each camera (USB, RTSP, or HTTP) is wrapped in a **Source Bin** containing:
- `v4l2src` (for USB cameras) or `uridecodebin` (for network streams)
=======
Each camera (USB, RTSP, HTTP, or HTTPS) is wrapped in a **Source Bin** containing:
- `v4l2src` (for USB cameras like `/dev/video0`)
- `uridecodebin` (for `rtsp://`, `http://`, `https://`, or `file://` streams)
>>>>>>> sharad
- `nvvideoconvert` — transfers raw frames into GPU memory (NVMM)
- `capsfilter` — enforces 640×480 NV12 format to keep GPU usage reasonable

### Stage 2: Stream Multiplexing
<<<<<<< HEAD
`nvstreammux` collects frames from **all** camera sources into a single batched buffer. This is critical because it allows the PGIE to process multiple camera frames in a single GPU inference pass, dramatically increasing throughput.
=======
`nvstreammux` collects frames from **all** camera sources into a single batched buffer. This allows the PGIE to process multiple camera frames in a single GPU inference pass.
>>>>>>> sharad

| Parameter | Value | Purpose |
|:---|:---|:---|
| `width` | 640 | Muxer output resolution |
| `height` | 480 | Muxer output resolution |
| `batch-size` | N (num cameras) | Frames batched per inference cycle |
| `batched-push-timeout` | 40000 µs | Max wait before pushing incomplete batch |

### Stage 3: Primary Inference (PGIE) — Face Detection
The **Primary GPU Inference Engine** runs YOLOv8n-face to detect all face bounding boxes in the batched frame.

- **Model**: `yolov8n-face.onnx` → TensorRT FP16 engine (auto-generated on first run)
<<<<<<< HEAD
- **Custom Parser**: `libnvds_infercustomparser_yolov8.so` — decodes YOLOv8's raw `[5, 8400]` output tensor into NvDsBBoxes
=======
- **Custom Parser**: `libnvds_infercustomparser_yolov8.so` — decodes YOLOv8's raw output tensor into NvDsBBoxes
>>>>>>> sharad
- **NMS**: Cluster mode 2 (Non-Maximum Suppression) with IoU threshold 0.45
- **Minimum Detection**: 32×32 pixels — filters out background noise and micro-artifacts

### Stage 4: Object Tracking
`nvtracker` with the IOU algorithm assigns **persistent integer IDs** to each detected face across frames. This is what enables the caching optimization — once a face is identified as "Sharad" on frame 10, we skip re-extracting the embedding on frames 11–160.

| Parameter | Value | Purpose |
|:---|:---|:---|
| `iouThreshold` | 0.2 | Lower = "stickier" tracking (fewer ID switches) |
| `minConsecutiveFrames` | 1 | Show box immediately without delay |
| `maxTargetsPerStream` | 20 | Max simultaneous tracked faces |

### Stage 5: Secondary Inference (SGIE) — Face Embedding
The **Secondary GPU Inference Engine** crops each detected face from the original frame, resizes it to **112×112** with symmetric center-padding, and runs MobileFaceNet to produce a **512-dimensional embedding vector**.

Key configuration parameters that ensure mathematical alignment:

| Parameter | Value | Effect |
|:---|:---|:---|
| `net-scale-factor` | 0.0078125 (1/128) | Normalization multiplier |
| `offsets` | 127.5;127.5;127.5 | Centering offset per channel |
| `symmetric-padding` | 1 | Center the face in the padded frame |
| `maintain-aspect-ratio` | 1 | Prevent distortion of facial geometry |
| `output-tensor-meta` | 1 | **Critical**: Attach raw tensor to metadata |
| `network-type` | 100 | "Other" — exposes raw tensor instead of classification |

### Stage 6: Python Probe (Application Logic)
A GStreamer pad probe is attached to the SGIE source pad. For every frame, it:

1. Iterates over all detected face objects
2. Checks the **Tracker ID Cache** for a previous match
3. On cache miss, extracts the 512-D tensor via `ctypes` pointer
4. Runs **vectorized cosine similarity** against the entire watchlist
5. Updates the OSD overlay (green box = recognised, red box = unknown)
6. Logs attendance to SQLite + CSV with deduplication

### Stage 7: Display
`nvmultistreamtiler` arranges all camera feeds into a grid, `nvdsosd` draws the bounding boxes and labels, and `nveglglessink` renders to the Jetson's HDMI display.

---

## 🧠 AI Models Used

### 1. YOLOv8n-face (Primary Detector)

| Property | Value |
|:---|:---|
| **Architecture** | YOLOv8 Nano (face-specific variant) |
| **Input** | 3×640×640 (RGB, normalized 0–1) |
<<<<<<< HEAD
| **Output** | `[batch, 5, 8400]` — (x, y, w, h, conf) × 8400 anchor boxes |
=======
| **Output** | Bounding boxes with confidence scores |
>>>>>>> sharad
| **Precision** | FP16 (TensorRT optimized) |
| **Custom Parser** | `NvDsInferParseYolo` in `libnvds_infercustomparser_yolov8.so` |
| **Source** | Exported from `yolov8n-face.pt` via Ultralytics |
| **ONNX Opset** | 11 (Jetson Nano compatibility) |

### 2. MobileFaceNet w600k (Embedding Extractor)

| Property | Value |
|:---|:---|
| **Architecture** | MobileFaceNet (InsightFace) |
| **Training Data** | WebFace600K — 600,000 identities |
| **Input** | 3×112×112 (RGB, normalized `(pixel - 127.5) / 128`) |
| **Output** | 512-dimensional L2-normalized embedding vector |
<<<<<<< HEAD
| **Output Layer** | `516` (auto-detected by TensorRT) |
=======
>>>>>>> sharad
| **Precision** | FP16 (TensorRT optimized) |
| **Source** | InsightFace model zoo (`w600k_mbf.onnx`) |

### 3. Haar Cascade (Enrollment-Only Detector)

| Property | Value |
|:---|:---|
| **Purpose** | Offline face cropping during enrollment |
| **Model** | `haarcascade_frontalface_default.xml` (bundled with OpenCV) |
| **Used In** | `enroll_trt.py` only — NOT in the live pipeline |

---

## ✨ Key Features

<<<<<<< HEAD
### 🔄 Multi-Camera Support
- Simultaneously processes **2+ camera feeds** (USB, RTSP, HTTP, or video files)
- Cameras can be placed in the same room for multi-angle coverage
- Cross-camera tracker-ID isolation prevents identity conflicts between cameras
=======
### 🔄 Multi-Camera & Multi-Protocol Support
- Simultaneously processes **2+ camera feeds** in a single pipeline
- Supported input types: **USB** (`/dev/video0`), **RTSP** (`rtsp://`), **HTTP** (`http://`), **HTTPS** (`https://`), **Video files** (`file://`)
- Cameras can be placed in the same room for multi-angle coverage
- Tiler automatically creates a side-by-side grid display
>>>>>>> sharad

### ⚡ Tracker Object-ID Cache
- Once a face is identified (e.g., "Sharad" with score 0.62), the result is **cached by tracker ID**
- Subsequent frames skip the entire SGIE inference + matching pipeline for that face
<<<<<<< HEAD
- Cache keys are prefixed with camera source ID: `"0_5"` vs `"1_5"` — so two cameras never collide
=======
>>>>>>> sharad
- Stale IDs are automatically evicted every 100 frames (configurable)
- Unknown faces are periodically retried every 5 frames for a second chance

### 📊 Vectorized Watchlist Matching
<<<<<<< HEAD
- All enrolled embeddings are pre-loaded into a single NumPy matrix at startup
- Live face matching is a single `matrix @ vector` operation (O(1) regardless of database size)
- Handles dynamic embedding dimensions (128-D or 512-D) with graceful truncation

### 📸 Multi-Angle Enrollment
- Enroll the same person with 3–5 photos from different angles
- Name format: `"Sharad_1"`, `"Sharad_2"`, `"Sharad_3"`
- Display automatically strips the `_N` suffix → shows "Sharad" on screen
- Dramatically improves recognition accuracy under varied head poses

### 📝 Dual Attendance Logging
- **SQLite** (`attendance.db`): Structured relational log with user/camera/timestamp
- **CSV** (`attendance.csv`): Excel-friendly format with `Name, Date, Time` columns
- **Deduplication**: 5-second cooldown per person prevents log spam
- **CSV Dedup Window**: 300-second (5-minute) window prevents duplicate CSV rows per class session
=======
- All enrolled embeddings are pre-loaded into a single NumPy matrix at startup: `shape = (N, 512)`
- Live face matching is a single `matrix @ vector` operation — **O(1) regardless of database size**
- Handles dynamic embedding dimensions (128-D or 512-D) with graceful truncation

### 📸 Multi-Angle Enrollment (Embedding Fusion)
- Place 3–5 photos per person (different angles) in `image_db/<name>/`
- `enroll_trt.py` automatically:
  1. Finds the face in each photo using Haar Cascades
  2. Extracts a 512-D embedding from each cropped face
  3. **Averages all embeddings** into a single fused vector
  4. L2-normalizes and stores the result in SQLite
- Dramatically improves recognition accuracy under varied head poses
- **Idempotent**: re-running enrollment safely skips already-enrolled students

### 📝 Dual Attendance Logging
- **SQLite** (`attendance.db`): Structured relational log with `user_id`, `camera_id`, `timestamp`
- **CSV** (`attendance.csv`): Excel-friendly format with `Name, Date, Time` columns
- **In-pipeline cooldown**: 5-second `COOLDOWN_SEC` per person prevents rapid re-logging
- **CSV dedup window**: 300-second (5-minute) window prevents duplicate CSV rows per class session
>>>>>>> sharad

### 🎨 Live OSD Overlay
- **Green box** + name + confidence score for recognised faces
- **Red box** + "Unknown" label for unrecognised faces
- **FPS counter** displayed on each camera stream
<<<<<<< HEAD
- Frame count and cache size printed to terminal every 30 frames
=======
- Display names automatically strip enrollment suffixes (e.g., `"Sharad_1"` → `"Sharad"`)
>>>>>>> sharad

---

## 📁 Project Structure

```
FACE_Detection_Jetson/
│
├── main_dual_cam.py          # 🎯 Main pipeline — run this to start
<<<<<<< HEAD
├── enroll_trt.py             # 📸 Enroll faces into the database
├── db_utils.py               # 🗄️  SQLite/CSV database operations
=======
├── enroll_trt.py             # 📸 Auto-enrollment from image_db/ folders
├── db_utils.py               # 🗄️  SQLite + CSV attendance logging with dedup
>>>>>>> sharad
├── fix_onnx.py               # 🔧 YOLO → ONNX export (run on PC, not Nano)
├── convert_models.py         # 🔧 Model conversion utility
│
├── configs/
│   ├── pgie_config.txt       # YOLO face detector configuration
│   ├── sgie_config.txt       # MobileFaceNet embedding configuration
│   └── tracker_config.yml    # IOU tracker parameters
│
├── models/
│   ├── yolov8n-face.onnx     # YOLO face detector (ONNX format)
│   └── w600k_mbf.onnx        # MobileFaceNet (ONNX format)
│                               # TensorRT .engine files auto-generated here
│
├── image_db/                 # 📂 Enrollment photos (organized by person)
│   ├── sharad/
│   │   ├── sharad_1.jpg
<<<<<<< HEAD
│   │   └── sharad_2.jpg
│   └── aditya/
│       └── aditya_1.jpg
=======
│   │   ├── sharad_2.jpg
│   │   └── sharad_3.jpg
│   ├── aditya/
│   │   ├── aditya_1.jpg
│   │   └── aditya_2.jpg
│   └── raj/
│       ├── raj_1.jpg
│       └── raj_2.jpg
>>>>>>> sharad
│
├── labels.txt                # Class labels for YOLO (just "face")
├── attendance.db             # SQLite database (auto-created)
├── attendance.csv            # CSV attendance log (auto-created)
│
├── libnvds_infercustomparser_yolov8.so  # Custom YOLO NMS parser (C++)
│
├── JETSON_SETUP.md           # Jetson-specific setup instructions
├── SETUP_GUIDE.md            # Quick start guide
└── README.md                 # This file
```

---

## 📄 File-by-File Breakdown

### `main_dual_cam.py` — The Main Pipeline

This is the entry point. It constructs and runs the entire GStreamer DeepStream pipeline.

**Key sections:**

<<<<<<< HEAD
| Section | Lines | Purpose |
|:---|:---|:---|
| Globals & Constants | 28–42 | Thresholds, cooldowns, cache settings |
| Watchlist Builder | 44–75 | Pre-loads DB embeddings into a NumPy matrix |
| `match_face_vectorized()` | 80–128 | Single matrix multiply for instant matching |
| `osd_sink_pad_buffer_probe()` | 133–330 | The probe function — all recognition logic |
| `_apply_recognised_overlay()` | 376–386 | Green box + name rendering |
| `_apply_unknown_overlay()` | 389–399 | Red box + "Unknown" rendering |
| `create_source_bin()` | 351–400 | Creates USB/RTSP/HTTP camera source bins |
| `main()` | 405–556 | Pipeline construction, linking, and execution |
=======
| Section | Purpose |
|:---|:---|
| Globals & Constants | Thresholds, cooldowns, cache settings |
| Watchlist Builder | Pre-loads DB embeddings into a NumPy matrix at startup |
| `match_face_vectorized()` | Single `matrix @ vector` for instant matching |
| `osd_sink_pad_buffer_probe()` | The probe function — all recognition + caching logic |
| `_apply_recognised_overlay()` | Green box + name (auto-strips `_N` suffix) |
| `_apply_unknown_overlay()` | Red box + "Unknown" rendering |
| `create_source_bin()` | Creates USB / RTSP / HTTP / HTTPS / file source bins |
| `main()` | Pipeline construction, linking, and GLib main loop |
>>>>>>> sharad

**How the probe processes each face:**

```
For each frame in the batch:
  For each detected face object:
    │
<<<<<<< HEAD
    ├── cache_key = f"{source_id}_{object_id}"
    │
    ├── CACHE HIT (known face)?
    │     ├── Recognised → Green overlay, log attendance
    │     └── Unknown → Red overlay, retry every 5 frames
    │
    └── CACHE MISS (new face)?
          ├── Extract 512-D tensor from SGIE metadata (via ctypes)
          ├── L2-normalize the embedding
          ├── scores = watchlist_matrix @ query_vector
          ├── best_match = argmax(scores)
          ├── Score > 0.40? → Recognised (cache + log + green box)
          └── Score ≤ 0.40? → Unknown (cache + red box)
```

### `enroll_trt.py` — Face Enrollment

Registers new faces into the database. Preprocessing **exactly matches** the SGIE's DeepStream preprocessing to ensure embedding consistency.

**Enrollment preprocessing pipeline:**
```
Raw Photo → Haar Cascade Face Crop → BGR→RGB → Symmetric Letterbox (112×112)
         → Normalize: (pixel - 127.5) / 128 → HWC→CHW → ONNX Inference → 512-D Vector
         → L2 Normalize → Store in SQLite
```

### `db_utils.py` — Database Layer

| Function | Purpose |
|:---|:---|
| `init_db()` | Creates `Users` and `Logs` tables if they don't exist |
| `save_embedding(name, embedding)` | INSERT or UPDATE a user's embedding BLOB |
| `load_all_embeddings()` | Returns all users as `[{user_id, name, embedding}]` |
| `log_attendance(user_id, camera_id)` | Inserts an attendance record with timestamp |

**Database Schema:**
```sql
Users (
    user_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT UNIQUE,
    embedding  BLOB    -- FP32 bytes, 512 × 4 = 2048 bytes per user
)

Logs (
    log_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER REFERENCES Users(user_id),
    camera_id  INTEGER,
    timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP
)
```

### `fix_onnx.py` — YOLO Export Script

**Run on your PC (not the Jetson Nano).**

Exports the YOLOv8n-face PyTorch model to ONNX format with specific settings required for DeepStream compatibility:
- `opset=11` — the only opset fully supported by TensorRT 8.x on Jetson Nano
- `dynamic=False` — forces static 640×640 input (required by DeepStream)
- `batch=16` — matches the PGIE `batch-size` for multi-stream throughput

---

## ⚙️ Configuration Deep Dive

### `configs/pgie_config.txt` — YOLOv8 Face Detector

```ini
[property]
onnx-file=models/yolov8n-face.onnx                    # Source ONNX model
model-engine-file=models/yolov8n-face.onnx_b16...     # Auto-generated TensorRT engine
network-mode=2                                          # FP16 precision
batch-size=16                                           # Max frames per GPU batch
interval=2                                              # Infer every 3rd frame (saves GPU)
parse-bbox-func-name=NvDsInferParseYolo                # Custom C++ bounding box parser
custom-lib-path=libnvds_infercustomparser_yolov8.so    # Compiled parser library

[class-attrs-all]
pre-cluster-threshold=0.15                              # Confidence floor
nms-iou-threshold=0.45                                  # NMS overlap threshold
topk=20                                                 # Max detections per frame
min-bbox-width=32                                       # Ignore tiny false positives
min-bbox-height=32
```

### `configs/sgie_config.txt` — MobileFaceNet Embeddings

```ini
[property]
operate-on-gie-id=1          # Process faces from PGIE (gie-unique-id=1)
operate-on-class-ids=0        # Process class 0 ("face") only
gie-unique-id=2               # This engine's identifier

# Preprocessing — MUST match enroll_trt.py exactly
net-scale-factor=0.0078125    # 1/128
offsets=127.5;127.5;127.5     # Per-channel subtraction
infer-dims=3;112;112          # Input dimensions (CHW)

network-type=100              # "Other" — exposes raw tensor output
output-tensor-meta=1          # CRITICAL: attaches embedding to object metadata
maintain-aspect-ratio=1       # Prevents face crop distortion
symmetric-padding=1           # Centers the face within the padded frame
```

### `configs/tracker_config.yml` — IOU Object Tracker

```yaml
IOUTracker:
  iouThreshold: 0.2                               # Low = stickier tracking
  minConsecutiveFrames: 1                          # Instant box display
  maxWaitFramesForSpatioTemporalConfidence: 3      # Frames before dropping lost ID
```

---

## 🔗 How Everything Connects

```
                     ┌──────────────────────────────┐
                     │     enroll_trt.py             │
                     │  (Run ONCE per student)       │
                     │                               │
                     │  Photo → HaarCascade Crop     │
                     │       → Preprocess (112×112)  │
                     │       → MobileFaceNet ONNX    │
                     │       → 512-D Embedding       │
                     └──────────┬───────────────────┘
                                │
                         save_embedding()
                                │
                                ▼
                     ┌──────────────────────────────┐
                     │       attendance.db            │
                     │  ┌───────────────────────┐    │
                     │  │ Users table            │    │
                     │  │ (name, embedding BLOB) │    │
                     │  └───────────┬───────────┘    │
                     │              │                 │
                     │  ┌───────────▼───────────┐    │
                     │  │ Logs table             │    │
                     │  │ (user_id, cam, time)   │    │
                     │  └───────────────────────┘    │
                     └──────────┬───────────────────┘
                                │
                     load_all_embeddings()
                                │
                                ▼
                     ┌──────────────────────────────┐
                     │     main_dual_cam.py           │
                     │  (Run to START the system)     │
                     │                               │
                     │  1. Load DB → Watchlist Matrix │
                     │  2. Build GStreamer Pipeline   │
                     │  3. Camera → PGIE → Tracker   │
                     │     → SGIE → Probe Function   │
                     │  4. Probe: extract tensor,     │
                     │     matrix multiply, overlay   │
                     │  5. log_attendance() on match  │
                     └──────────────────────────────┘
```

### The Critical Alignment Chain

For accurate recognition, the preprocessing in `enroll_trt.py` (offline) **must be byte-for-byte identical** to what DeepStream's SGIE does (live). Here is how they align:

| Step | SGIE Config (Live) | enroll_trt.py (Offline) |
|:---|:---|:---|
| Color Space | `model-color-format=0` (BGR→RGB) | `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` |
| Resize | `infer-dims=3;112;112` | `cv2.resize(img, (new_w, new_h))` |
| Padding | `symmetric-padding=1` | `cv2.copyMakeBorder(..., BORDER_CONSTANT)` |
| Normalization | `(pixel - 127.5) × 0.0078125` | `(pixel - 127.5) / 128.0` |
| Output | 512-D FP16 tensor | 512-D FP32 tensor → L2 normalized |

---

## 🚀 Setup & Installation

### Prerequisites (on Jetson Nano)

- **JetPack 4.6** (L4T 32.6.1)
- **DeepStream SDK 6.0.1**
- **Python 3.6** with GStreamer bindings (`pyds`)
- **OpenCV** (included with JetPack)
- **ONNX Runtime** (`pip3 install onnxruntime`)
- **NumPy** (`pip3 install numpy`)

### Step 1: Clone & Prepare

```bash
cd /home/blackbox
git clone <repository_url> FACE_Detection_Jetson
cd FACE_Detection_Jetson
```

### Step 2: Place Model Files

Copy the following model files into the `models/` directory:

```bash
models/
├── yolov8n-face.onnx     # YOLOv8 face detector
└── w600k_mbf.onnx        # InsightFace MobileFaceNet w600k
```

> **Note:** The ONNX models must be converted on your PC first using `fix_onnx.py` (for YOLO) and then transferred to the Jetson Nano. TensorRT `.engine` files are generated automatically on the Nano's first run.

### Step 3: Compile the Custom YOLO Parser (if not already provided)

```bash
# Only needed if libnvds_infercustomparser_yolov8.so is not present
cd /opt/nvidia/deepstream/deepstream/sources/libs/nvdsinfer_customparser
make
cp libnvds_infercustomparser_yolov8.so /home/blackbox/FACE_Detection_Jetson/
```

### Step 4: Initialize the Database

```bash
python3 db_utils.py
# Output: "Database initialized successfully."
```

=======
    ├── CACHE HIT (tracker ID already identified)?
    │     ├── Recognised → Green overlay, log attendance (with cooldown)
    │     └── Unknown → Red overlay, retry SGIE every 5 frames
    │
    └── CACHE MISS (new face)?
          ├── Filter by tensor_meta.unique_id == 2 (SGIE only)
          ├── Extract 512-D tensor from SGIE metadata (via ctypes)
          ├── L2-normalize the live embedding
          ├── scores = watchlist_matrix @ query_vector    ← single operation
          ├── best_match = argmax(scores)
          ├── Score > 0.40? → Recognised (cache + log + green box)
          └── Score ≤ 0.40? → Unknown (cache + red box)

  Every 100 frames: Evict stale cache entries (not seen for 150+ frames)
```

### `enroll_trt.py` — Face Enrollment

Registers new faces into the database. Supports **two modes**:

**Auto-scan mode** (recommended):
```bash
python3 enroll_trt.py    # Scans all folders in image_db/
```
- Discovers all student folders under `image_db/`
- For each student: extracts face crops from **all** images, averages their embeddings, and stores a single fused vector
- **Idempotent**: already-enrolled students are completely skipped

**Legacy single-image mode:**
```bash
python3 enroll_trt.py sharad.jpg "Sharad"
```

**Enrollment preprocessing pipeline (must match SGIE exactly):**
```
Raw Photo → Haar Cascade Face Crop → BGR→RGB → Symmetric Letterbox (112×112)
         → Normalize: (pixel - 127.5) / 128 → HWC→CHW → ONNX Inference → 512-D
         → L2 Normalize → Average across all angles → Store in SQLite
```

### `db_utils.py` — Database Layer

| Function | Purpose |
|:---|:---|
| `init_db()` | Creates `Users` and `Logs` tables if they don't exist |
| `has_embedding(name)` | Check if student already enrolled (for idempotency) |
| `save_embedding(name, embedding)` | INSERT or UPDATE a user's FP32 embedding BLOB |
| `load_all_embeddings()` | Returns all users as `[{user_id, name, embedding}]` |
| `log_attendance(user_id, camera_id, student_name)` | Logs to SQLite + CSV with dedup |
| `_is_duplicate_csv_entry(name, now)` | 300-second sliding window deduplication |

**Database Schema:**
```sql
Users (
    user_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT UNIQUE,
    embedding  BLOB    -- FP32 bytes, 512 × 4 = 2048 bytes per user
)

Logs (
    log_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER REFERENCES Users(user_id),
    camera_id  INTEGER,
    timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP
)
```

**CSV Format** (`attendance.csv`):
```
Name,Date,Time
Sharad,2026-04-19,10:05:32
Aditya,2026-04-19,10:05:45
```

### `fix_onnx.py` — YOLO Export Script

**Run on your PC (not the Jetson Nano).**

Exports the YOLOv8n-face PyTorch model to ONNX format with specific settings required for DeepStream compatibility:
- `opset=11` — the only opset fully supported by TensorRT 8.x on Jetson Nano
- `dynamic=False` — forces static 640×640 input (required by DeepStream)
- `batch=16` — matches the PGIE `batch-size` for multi-stream throughput

---

## ⚙️ Configuration Deep Dive

### `configs/pgie_config.txt` — YOLOv8 Face Detector

```ini
[property]
onnx-file=models/yolov8n-face.onnx                    # Source ONNX model
model-engine-file=models/yolov8n-face.onnx_b16...     # Auto-generated TensorRT engine
network-mode=2                                          # FP16 precision
batch-size=16                                           # Max frames per GPU batch
interval=2                                              # Infer every 3rd frame (saves GPU)
parse-bbox-func-name=NvDsInferParseYolo                # Custom C++ bounding box parser
custom-lib-path=libnvds_infercustomparser_yolov8.so    # Compiled parser library

[class-attrs-all]
pre-cluster-threshold=0.15                              # Confidence floor
nms-iou-threshold=0.45                                  # NMS overlap threshold
topk=20                                                 # Max detections per frame
min-bbox-width=32                                       # Ignore tiny false positives
min-bbox-height=32
```

### `configs/sgie_config.txt` — MobileFaceNet Embeddings

```ini
[property]
operate-on-gie-id=1          # Process faces from PGIE (gie-unique-id=1)
operate-on-class-ids=0        # Process class 0 ("face") only
gie-unique-id=2               # This engine's identifier

# Preprocessing — MUST match enroll_trt.py exactly
net-scale-factor=0.0078125    # 1/128
offsets=127.5;127.5;127.5     # Per-channel subtraction
model-color-format=0          # BGR→RGB conversion
infer-dims=3;112;112          # Input dimensions (CHW)
symmetric-padding=1           # Centers the face within the padded frame
maintain-aspect-ratio=1       # Prevents face crop distortion

network-type=100              # "Other" — exposes raw tensor output
output-tensor-meta=1          # CRITICAL: attaches embedding to object metadata
batch-size=1                  # Face crops processed per inference cycle
```

### `configs/tracker_config.yml` — IOU Object Tracker

```yaml
IOUTracker:
  iouThreshold: 0.2                               # Low = stickier tracking
  minConsecutiveFrames: 1                          # Instant box display
  maxWaitFramesForSpatioTemporalConfidence: 3      # Frames before dropping lost ID
```

---

## 🔗 How Everything Connects

```
                     ┌──────────────────────────────┐
                     │     enroll_trt.py             │
                     │  (Run ONCE per student)       │
                     │                               │
                     │  image_db/sharad/*.jpg        │
                     │     → HaarCascade Crop        │
                     │     → Symmetric Pad (112×112) │
                     │     → w600k_mbf.onnx          │
                     │     → 512-D Embedding ×N      │
                     │     → Average + L2 Normalize  │
                     └──────────┬───────────────────┘
                                │
                         save_embedding()
                                │
                                ▼
                     ┌──────────────────────────────┐
                     │       attendance.db            │
                     │  ┌───────────────────────┐    │
                     │  │ Users table            │    │
                     │  │ (name, embedding BLOB) │    │
                     │  └───────────┬───────────┘    │
                     │              │                 │
                     │  ┌───────────▼───────────┐    │
                     │  │ Logs table             │    │
                     │  │ (user_id, cam, time)   │    │
                     │  └───────────────────────┘    │
                     └──────────┬───────────────────┘
                                │
                     load_all_embeddings()  →  Watchlist Matrix (N, 512)
                                │
                                ▼
                     ┌──────────────────────────────┐
                     │     main_dual_cam.py           │
                     │  (Run to START the system)     │
                     │                               │
                     │  1. Load DB → Watchlist Matrix │
                     │  2. Build GStreamer Pipeline   │
                     │  3. Camera → PGIE → Tracker   │
                     │     → SGIE → Probe Function   │
                     │  4. Probe: extract tensor,     │
                     │     matrix @ query, overlay   │
                     │  5. log_attendance() on match  │
                     │     → SQLite + CSV (deduped)  │
                     └──────────────────────────────┘
```

### The Critical Alignment Chain

For accurate recognition, the preprocessing in `enroll_trt.py` (offline) **must be identical** to what DeepStream's SGIE does (live). Here is how they align:

| Step | SGIE Config (Live) | enroll_trt.py (Offline) |
|:---|:---|:---|
| Color Space | `model-color-format=0` (BGR→RGB) | `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` |
| Resize | `infer-dims=3;112;112` | `cv2.resize(img, (new_w, new_h))` |
| Padding | `symmetric-padding=1` | `cv2.copyMakeBorder(..., BORDER_CONSTANT)` |
| Normalization | `(pixel - 127.5) × 0.0078125` | `(pixel - 127.5) / 128.0` |
| Output | 512-D FP16 tensor | 512-D FP32 tensor → L2 normalized |

---

## 🚀 Setup & Installation

### Prerequisites (on Jetson Nano)

- **JetPack 4.6.1** (L4T 32.7.1)
- **DeepStream SDK 6.0.1**
- **Python 3.6** with GStreamer bindings (`pyds`)
- **OpenCV** (included with JetPack)
- **ONNX Runtime** (`pip3 install onnxruntime`)
- **NumPy** (`pip3 install numpy`)

### Step 1: Clone & Prepare

```bash
cd /home/blackbox
git clone <repository_url> FACE_Detection_Jetson
cd FACE_Detection_Jetson
```

### Step 2: Place Model Files

Copy the following model files into the `models/` directory:

```bash
models/
├── yolov8n-face.onnx     # YOLOv8 face detector
└── w600k_mbf.onnx        # InsightFace MobileFaceNet w600k
```

> **Note:** TensorRT `.engine` files are generated automatically on the Nano's first run (takes ~3–5 minutes). They are then cached permanently.

### Step 3: Initialize the Database

```bash
python3 db_utils.py
# Output: "Database and CSV initialized successfully."
```

### Step 4: Set Up Enrollment Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy opencv-python onnxruntime
deactivate
```

>>>>>>> sharad
---

## 📖 Usage Guide

### 🔹 Enrolling Faces

<<<<<<< HEAD
**Single image enrollment:**
```bash
python3 enroll_trt.py sharad.jpg "Sharad"
```

**Multi-angle enrollment (recommended for higher accuracy):**
```bash
python3 enroll_trt.py sharad_front.jpg "Sharad_1"
python3 enroll_trt.py sharad_left.jpg  "Sharad_2"
python3 enroll_trt.py sharad_right.jpg "Sharad_3"
```

> The `_1`, `_2`, `_3` suffixes are automatically stripped in the display. All angles will show as **"Sharad"** on screen while the system picks whichever angle produces the highest match score.

### 🔹 Running the Live Pipeline

**Single USB camera:**
```bash
=======
**Step 1: Organize photos in `image_db/`:**
```
image_db/
├── sharad/
│   ├── sharad_front.jpg
│   ├── sharad_left.jpg
│   └── sharad_right.jpg
├── aditya/
│   ├── aditya_1.jpg
│   └── aditya_2.jpg
└── raj/
    └── raj_1.jpg
```

**Step 2: Run auto-enrollment (in venv):**
```bash
source venv/bin/activate
python3 enroll_trt.py
deactivate
```

> **Idempotent & safe to re-run:** Already-enrolled students are completely skipped.

**Legacy single-image enrollment:**
```bash
python3 enroll_trt.py sharad.jpg "Sharad"
```

### 🔹 Running the Live Pipeline

> ⚠️ **Must use SYSTEM Python (no venv)** — DeepStream's `pyds` bindings are installed system-wide.

```bash
deactivate 2>/dev/null    # Exit venv if active

# Single USB camera
>>>>>>> sharad
python3 main_dual_cam.py /dev/video0

# Dual USB cameras (same room, different angles)
python3 main_dual_cam.py /dev/video0 /dev/video1

# RTSP network camera
python3 main_dual_cam.py rtsp://admin:password@192.168.1.100:554/stream1

# HTTP camera (e.g., Android IP Webcam app)
python3 main_dual_cam.py http://192.168.1.104:8080/video

# HTTPS camera
python3 main_dual_cam.py https://192.168.1.104:8443/video

# Mixed sources (USB + Network)
python3 main_dual_cam.py /dev/video0 http://192.168.1.104:8080/video

# Pre-recorded video file
python3 main_dual_cam.py file:///home/blackbox/test_video.mp4
```

### 🔹 Viewing Attendance Logs

**CSV (open in Excel/LibreOffice):**
```bash
cat attendance.csv
# Name,Date,Time
# Sharad,2026-04-19,10:05:32
# Aditya,2026-04-19,10:05:45
```

**SQLite (programmatic):**
```bash
sqlite3 attendance.db "SELECT Users.name, Logs.camera_id, Logs.timestamp FROM Logs JOIN Users ON Logs.user_id = Users.user_id ORDER BY Logs.timestamp DESC LIMIT 20;"
```

### 🔹 Debugging (Verbose GStreamer Logs)

If faces are detected but not producing embeddings:
```bash
GST_DEBUG=3 python3 main_dual_cam.py /dev/video0
```

<<<<<<< HEAD
**Dual USB cameras (same room, different angles):**
```bash
python3 main_dual_cam.py /dev/video0 /dev/video1
```

**RTSP network camera:**
```bash
python3 main_dual_cam.py rtsp://admin:password@192.168.1.100:554/stream1
```

**HTTP camera (e.g., Android IP Webcam app):**
```bash
python3 main_dual_cam.py http://192.168.1.104:8080/video
```

**Mixed sources (USB + Network):**
```bash
python3 main_dual_cam.py /dev/video0 rtsp://192.168.1.100:554/stream1
```

**Testing with a pre-recorded video file:**
```bash
python3 main_dual_cam.py file:///home/blackbox/test_video.mp4
```

### 🔹 Viewing Attendance Logs

**SQLite (programmatic):**
```bash
sqlite3 attendance.db "SELECT Users.name, Logs.camera_id, Logs.timestamp FROM Logs JOIN Users ON Logs.user_id = Users.user_id ORDER BY Logs.timestamp DESC LIMIT 20;"
```

**CSV (open in Excel/LibreOffice):**
```bash
cat attendance.csv
# Name, Date, Time
# Sharad, 2026-04-19, 10:05:32
# Aditya, 2026-04-19, 10:05:45
```

### 🔹 Debugging (Verbose GStreamer Logs)

If faces are detected but not producing embeddings:
```bash
GST_DEBUG=3 python3 main_dual_cam.py /dev/video0
```

---

## ⚡ Performance Tuning

### Tunable Parameters in `main_dual_cam.py`

| Constant | Default | Effect |
|:---|:---|:---|
| `SIMILARITY_THRESHOLD` | 0.40 | Lower = more lenient matching, higher = stricter |
| `COOLDOWN_SEC` | 5 | Seconds between re-logging same person |
| `CACHE_EVICT_INTERVAL` | 100 | Frames between stale cache cleanup |
| `CACHE_STALE_FRAMES` | 150 | Evict tracker IDs not seen for this many frames |

### PGIE Tuning (`pgie_config.txt`)

| Parameter | Default | Tuning Advice |
|:---|:---|:---|
| `interval` | 2 | Higher = fewer GPU cycles (faster), but slower detection |
| `pre-cluster-threshold` | 0.15 | Raise to 0.25 if too many false positives |
| `topk` | 20 | Lower to 5 if you expect few faces |

### Understanding Similarity Scores

| Score Range | Meaning |
|:---|:---|
| 0.00 – 0.15 | Completely different people |
| 0.15 – 0.35 | Slight incidental resemblance |
| 0.35 – 0.45 | Uncertain — edge of threshold |
| **0.45 – 0.70** | **Same person (normal operating range)** |
| 0.70 – 0.85 | Very high confidence (ideal lighting/angle) |
| 0.85+ | Near-identical image (rare in live scenarios) |

---

## 🔧 Troubleshooting

### "SGIE skipped" / No embeddings

1. Verify `operate-on-class-ids=0` is set in `sgie_config.txt`
2. Verify `output-tensor-meta=1` is set in `sgie_config.txt`
3. Check that the engine file path in `model-engine-file` matches the actual batch size (e.g., `_b1_` for `batch-size=1`)
4. Delete stale `.engine` files and let TensorRT rebuild: `rm models/*.engine`

### Engine rebuilds every time

- Ensure `model-engine-file` filename contains the correct batch size suffix
- Example: `batch-size=1` → filename must contain `_b1_`

### Low match scores (<0.40)

1. Re-enroll using the latest `enroll_trt.py` (with symmetric padding)
2. Use multi-angle enrollment (3–5 photos per person)
3. Ensure enrollment photos are well-lit, front-facing, and tightly cropped

### Pipeline crashes / GPU freeze

1. Reduce `topk` in `pgie_config.txt` to limit detections
2. Raise `pre-cluster-threshold` to 0.25
3. Increase `interval` to 3 or 4 to reduce GPU load
4. Use 640×480 resolution (not 1280×720)

---

## 📄 License

This project is for educational and internal use.

---

=======
---

## ⚡ Performance Tuning

### Tunable Parameters in `main_dual_cam.py`

| Constant | Default | Effect |
|:---|:---|:---|
| `SIMILARITY_THRESHOLD` | 0.40 | Lower = more lenient matching, higher = stricter |
| `COOLDOWN_SEC` | 5 | Seconds between re-logging same person |
| `CACHE_EVICT_INTERVAL` | 100 | Frames between stale cache cleanup |
| `CACHE_STALE_FRAMES` | 150 | Evict tracker IDs not seen for this many frames |

### PGIE Tuning (`pgie_config.txt`)

| Parameter | Default | Tuning Advice |
|:---|:---|:---|
| `interval` | 2 | Higher = fewer GPU cycles (faster), but slower detection |
| `pre-cluster-threshold` | 0.15 | Raise to 0.25 if too many false positives |
| `topk` | 20 | Lower to 5 if you expect few faces |

### Understanding Similarity Scores

| Score Range | Meaning |
|:---|:---|
| 0.00 – 0.15 | Completely different people |
| 0.15 – 0.35 | Slight incidental resemblance |
| 0.35 – 0.45 | Uncertain — edge of threshold |
| **0.45 – 0.70** | **Same person (normal operating range)** |
| 0.70 – 0.85 | Very high confidence (ideal lighting/angle) |
| 0.85+ | Near-identical image (rare in live scenarios) |

### Performance Expectations (Jetson Nano 4GB)

| Metric | Before (v1) | After (v2) |
|:---|:---|:---|
| Python math/frame (1 face) | Every frame | First frame only |
| Python math/frame (3 faces) | 3× every frame | 3× on first appearance only |
| NumPy ops per match | N dot products (loop) | 1 matrix multiply |
| Multi-face accuracy | Broken (garbage tensors) | Correct 1:1 mapping |
| Expected FPS (1 cam, 2 faces) | ~10–14 | ~18–24 |

---

## 🔧 Troubleshooting

### "SGIE skipped" / No embeddings

1. Verify `operate-on-class-ids=0` is set in `sgie_config.txt`
2. Verify `output-tensor-meta=1` is set in `sgie_config.txt`
3. Check that the engine file path in `model-engine-file` matches the actual batch size (e.g., `_b1_` for `batch-size=1`)
4. Delete stale `.engine` files and let TensorRT rebuild: `rm models/*.engine`

### Engine rebuilds every time

- Ensure `model-engine-file` filename contains the correct batch size suffix
- Example: `batch-size=1` → filename must contain `_b1_`

### "Unknown" on all faces

1. Check enrollment: `python3 -c "from db_utils import load_all_embeddings; e=load_all_embeddings(); print(len(e), 'faces')"`
2. Ensure enrollment and SGIE use the **same** model (`w600k_mbf.onnx`)
3. Lower threshold: try `SIMILARITY_THRESHOLD = 0.30`

### Low match scores (<0.40)

1. Re-enroll using the latest `enroll_trt.py` (with symmetric padding)
2. Use multi-angle enrollment (3–5 photos per person)
3. Ensure enrollment photos are well-lit, front-facing, and tightly cropped

### Pipeline crashes / GPU freeze

1. Reduce `topk` in `pgie_config.txt` to limit detections
2. Raise `pre-cluster-threshold` to 0.25
3. Increase `interval` to 3 or 4 to reduce GPU load
4. Use 640×480 resolution (not 1280×720)

### Low FPS

1. Increase PGIE `interval` (e.g., 2 → 4)
2. The tracker cache dramatically reduces Python overhead
3. Consider reducing tiler resolution from 1280×720 to 960×540

---

>>>>>>> sharad
*Built for NVIDIA Jetson Nano · DeepStream 6.0.1 · Python GStreamer*
