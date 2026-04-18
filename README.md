# 🎯 Real-Time Face Recognition Attendance System

**Platform:** NVIDIA Jetson Nano 4GB  
**Stack:** JetPack 4.6.1 · DeepStream 6.0.1 · CUDA 10.2 · TensorRT 8.2  
**Pipeline:** GStreamer (Python bindings via `pyds`)

---

## Architecture Overview

```
┌─────────────┐
│  USB Camera  │
│ /dev/video0  │
└──────┬──────┘
       │ v4l2src
       ▼
  nvvideoconvert
   (→ NVMM NV12)
       │
       ▼
┌───────────────┐
│ nvstreammux   │  batch-size=1, 640×480
└───────┬───────┘
        ▼
┌───────────────┐
│  PGIE         │  YOLOv8n-face (TensorRT FP16)
│  (Detector)   │  1-class output, custom bbox parser
└───────┬───────┘
        ▼
┌───────────────┐
│  nvtracker    │  IOU low-level tracker
│  (Tracker)    │  Assigns persistent object_id
└───────┬───────┘
        ▼
┌───────────────┐
│  SGIE         │  MobileFaceNet w600k (TensorRT FP16)
│  (Recognizer) │  112×112 input → 512-D embedding
└───────┬───────┘
        │
┌───────┴──────────── Probe (Python) ──────────────┐
│                                                   │
│  1. Check track_id_cache[object_id]               │
│     → HIT:  reuse cached name (skip tensor math)  │
│     → MISS: extract tensor, matrix @ query,       │
│             store result in cache                  │
│                                                   │
│  2. Apply overlay (green=recognised, red=unknown)  │
│  3. Log attendance with cooldown                   │
│  4. Evict stale cache entries every 100 frames     │
└──────────────────────┬───────────────────────────┘
                       ▼
               nvmultistreamtiler
                  (1280×720)
                       ▼
                   nvdsosd
                       ▼
                nveglglessink
                 (sync=False)
```

> **Multi-camera:** The pipeline supports multiple cameras. Just pass additional
> device paths: `python3 main_dual_cam.py /dev/video0 /dev/video1`

---

## Key Features

### 1. Tracker Object-ID Cache
```python
track_id_cache = {object_id: (name, score, last_frame_seen)}
```
- When a face is **first detected**, the tracker assigns it a unique `object_id`
- On the **first frame**, the SGIE tensor is extracted and cosine similarity is computed
- On **every subsequent frame**, the cached result is reused — **zero Python math**
- Stale entries are evicted every 100 frames (configurable via `CACHE_EVICT_INTERVAL`)

### 2. Vectorized Watchlist Matching
```python
# At startup: build (N, D) matrix from all enrolled embeddings
watchlist_matrix = np.vstack([...])   # shape: (N, 512)

# At runtime: single dot product
scores = watchlist_matrix @ query     # shape: (N,)
best = np.argmax(scores)              # O(1) vs O(N) loop
```
- All DB embeddings are pre-normalized at startup
- Single `matrix @ vector` computes all cosine similarities simultaneously
- Handles dimension mismatches (512-D vs 128-D) by auto-truncating

### 3. Multi-Face SGIE Fix
**Root Cause:** `batch-size=1` in SGIE config meant only one face crop was inferred per batch. Additional faces received stale/garbage tensor pointers.

**Fix applied to `sgie_config.txt`:**

| Setting | Before | After | Why |
|---|---|---|---|
| `batch-size` | 1 | 16 | Process up to 16 face crops per batch |
| `maintain-aspect-ratio` | _(missing)_ | 1 | Prevent face crop distortion |
| `input-object-min-width` | 16 | 32 | Avoid garbage micro-crops |
| `input-object-min-height` | 16 | 32 | Avoid garbage micro-crops |

The probe also filters tensor metadata by `tensor_meta.unique_id == 2` (SGIE's GIE ID) to prevent reading PGIE tensors by mistake.

---

## Project Structure

```
FACE_Detection_Jetson/
├── main_dual_cam.py          # Main DeepStream pipeline (run with SYSTEM Python)
├── db_utils.py               # SQLite helpers (Users table, Logs table)
├── enroll_trt.py             # Offline face enrollment (run in venv)
├── fix_onnx.py               # YOLOv8 ONNX compatibility fixer (if re-export needed)
├── labels.txt                # Single class label: "face"
├── attendance.db             # SQLite database (auto-created)
│
├── configs/
│   ├── pgie_config.txt       # YOLOv8n-face detector config
│   ├── sgie_config.txt       # MobileFaceNet recognizer config
│   └── tracker_config.yml    # IOU tracker config
│
├── models/                   # ← Already on Jetson
│   ├── yolov8n-face.onnx
│   ├── yolov8n-face.onnx_b1_gpu0_fp16.engine
│   ├── w600k_mbf.onnx
│   └── w600k_mbf.onnx_b16_gpu0_fp16.engine
│
├── libnvds_infercustomparser_yolov8.so   # Custom YOLO bbox parser
├── JETSON_SETUP.md           # Comprehensive Jetson setup & next-steps
└── README.md                 # This file
```

---

## Quick Start

### 1. Enroll Faces (in venv)
```bash
source venv/bin/activate
python3 enroll_trt.py sharad.jpg "Sharad"
python3 enroll_trt.py aditya.jpg "Aditya"
python3 enroll_trt.py RAJ.jpg "Raj"
deactivate
```

### 2. Run Pipeline (SYSTEM Python — no venv!)
```bash
deactivate 2>/dev/null
python3 main_dual_cam.py /dev/video0
```

---

## Configuration Reference

### Tunable Constants (in `main_dual_cam.py`)

| Constant | Default | Description |
|---|---|---|
| `SIMILARITY_THRESHOLD` | 0.40 | Cosine similarity floor for positive ID |
| `COOLDOWN_SEC` | 5 | Seconds between re-logging same person |
| `CACHE_EVICT_INTERVAL` | 100 | Frames between stale-ID cleanup passes |
| `CACHE_STALE_FRAMES` | 150 | Evict IDs not seen for this many frames |

### PGIE (YOLOv8n-face) — `configs/pgie_config.txt`

| Key Setting | Value | Notes |
|---|---|---|
| `network-mode` | 2 | FP16 inference |
| `interval` | 4 | Run detector every 4th frame (saves GPU) |
| `pre-cluster-threshold` | 0.15 | Detection confidence floor |
| `nms-iou-threshold` | 0.45 | NMS overlap threshold |
| `topk` | 20 | Max detections per frame |

### SGIE (MobileFaceNet) — `configs/sgie_config.txt`

| Key Setting | Value | Notes |
|---|---|---|
| `batch-size` | 16 | Max face crops per inference batch |
| `maintain-aspect-ratio` | 1 | Prevent crop distortion |
| `output-tensor-meta` | 1 | Attach raw tensor for Python extraction |
| `infer-dims` | 3;112;112 | MobileFaceNet input shape |
| `network-type` | 100 | "Other" (not classifier/detector) |

---

## Troubleshooting

### "Unknown" on all faces
1. **Check enrollment:** `python3 -c "from db_utils import load_all_embeddings; e=load_all_embeddings(); print(len(e), 'faces')"`
2. **Check dimensions match:** enrollment and SGIE must use the same model (`w600k_mbf.onnx`)
3. **Lower threshold:** try `SIMILARITY_THRESHOLD = 0.30`

### SGIE not firing (no tensor metadata)
1. Verify `operate-on-gie-id=1` matches PGIE's `gie-unique-id=1`
2. Verify `operate-on-class-ids=0` matches the face class ID
3. Check `input-object-min-width/height` — face detection box must be ≥32px

### TensorRT engine rebuild
If you change `batch-size` in SGIE config, **delete the old engine file** so TensorRT rebuilds:
```bash
rm models/w600k_mbf.onnx_b1_gpu0_fp16.engine
```

### Low FPS
1. Increase PGIE `interval` (e.g., 4 → 6)
2. The tracker cache dramatically reduces Python overhead
3. Consider reducing tiler resolution from 1280×720 to 960×540

---

## Performance Expectations (Jetson Nano 4GB)

| Metric | Before (v1) | After (v2) |
|---|---|---|
| Python math/frame (1 face) | Every frame | First frame only |
| Python math/frame (3 faces) | 3× every frame | 3× on first appearance only |
| NumPy ops per match | N dot products (loop) | 1 matrix multiply |
| Multi-face accuracy | Broken (garbage tensors) | Correct 1:1 mapping |
| Expected FPS (1 cam, 2 faces) | ~10-14 | ~18-24 |
