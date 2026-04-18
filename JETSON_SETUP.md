# 🔧 Jetson Nano Setup & Continuation Guide

> **Context:** You're developing on Windows. This guide is for when you sit down
> at your Jetson Nano and need to get everything running end-to-end.

---

## ✅ What Has Been Achieved (on Windows)

| Component | Status | Details |
|---|---|---|
| `main_dual_cam.py` | ✅ Rewritten | Tracker cache, vectorized matching, multi-face fix |
| `configs/sgie_config.txt` | ✅ Updated | batch-size=16, maintain-aspect-ratio=1 |
| `configs/pgie_config.txt` | ✅ Ready | YOLOv8n-face with custom parser |
| `configs/tracker_config.yml` | ✅ Ready | IOU tracker, low GPU usage |
| `db_utils.py` | ✅ Ready | SQLite Users + Logs tables |
| `enroll_trt.py` | ✅ Ready | Offline enrollment via ONNX Runtime |
| `labels.txt` | ✅ Ready | Single "face" class |
| `README.md` | ✅ Created | Full architecture docs |

### Already on Jetson (no action needed)

| File | Location on Jetson |
|---|---|
| `yolov8n-face.onnx` + `.engine` | `models/` |
| `w600k_mbf.onnx` + `.engine` | `models/` |
| `libnvds_infercustomparser_yolov8.so` | project root |

---

## 🔴 What Needs to Be Done on the Jetson Nano

### Phase 1: System Prerequisites (One-Time Verification)

These should already be present if you flashed JetPack 4.6.1. Run to confirm:

```bash
# ---- Verify JetPack & DeepStream ---- #
cat /etc/nv_tegra_release
# Expected: R32 (release), REVISION: 7.1 (or similar for JP 4.6.1)

dpkg -l | grep deepstream
# Expected: deepstream-6.0  6.0.1-1  arm64

nvcc --version
# Expected: Cuda compilation tools, release 10.2

python3 -c "import pyds; print('pyds OK')"
# Expected: pyds OK

python3 -c "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst; print('GStreamer OK')"
# Expected: GStreamer OK
```

If `pyds` is NOT found, install the DeepStream Python bindings:
```bash
cd /opt/nvidia/deepstream/deepstream/lib
pip3 install pyds-1.1.1-py3-none*.whl
```

### Phase 2: Transfer Updated Code to Jetson

```bash
# Option A: Git pull (if using a repo)
cd /home/blackbox/FACE_Detection_Jetson
git pull

# Option B: SCP from Windows
scp -r user@windows-ip:e:/jetson_nano/* /home/blackbox/FACE_Detection_Jetson/

# Option C: USB drive
cp -r /media/usb/jetson_nano/* /home/blackbox/FACE_Detection_Jetson/
```

> **Important:** Only copy the Python/config/doc files. Do NOT overwrite the
> `models/` directory — your ONNX and engine files are already there.

### Phase 3: Delete Old SGIE Engine (Required!)

Since we changed SGIE `batch-size` from 1 → 16, the old engine is incompatible.
TensorRT will rebuild it automatically on first run (~5 min).

```bash
cd /home/blackbox/FACE_Detection_Jetson/models

# Delete ONLY the old SGIE engine (the PGIE engine is unchanged)
rm -f w600k_mbf.onnx_b1_gpu0_fp16.engine

echo "Old SGIE engine deleted. TensorRT will rebuild with batch-size=16 on first run."
```

> If your SGIE engine filename is different, delete any `w600k_mbf*.engine` file.

### Phase 4: Virtual Environment (for enrollment scripts only)

```bash
cd /home/blackbox/FACE_Detection_Jetson

# Create venv with system-site-packages (inherits pyds, gi, etc.)
python3 -m venv venv --system-site-packages
source venv/bin/activate

# Install enrollment dependencies
pip install numpy opencv-python onnxruntime

# Verify
python3 -c "import numpy; import cv2; import onnxruntime; print('venv OK')"
```

### Phase 5: Enroll Faces

```bash
cd /home/blackbox/FACE_Detection_Jetson
source venv/bin/activate

# Initialize the database
python3 -c "from db_utils import init_db; init_db(); print('DB ready')"

# Enroll each person (one clear face photo per person)
python3 enroll_trt.py sharad.jpg "Sharad"
python3 enroll_trt.py aditya.jpg "Aditya"
python3 enroll_trt.py RAJ.jpg "Raj"

# Verify enrollments
python3 -c "
from db_utils import load_all_embeddings
users = load_all_embeddings()
for u in users:
    print(f\"  {u['name']}  dim={len(u['embedding'])}  user_id={u['user_id']}\")
print(f'Total: {len(users)} enrolled faces')
"

deactivate  # IMPORTANT: deactivate venv before running pipeline
```

### Phase 6: Run the Pipeline (SYSTEM Python!)

```bash
cd /home/blackbox/FACE_Detection_Jetson

# ⚠️  MUST use system Python — NOT the venv ⚠️
deactivate 2>/dev/null

# Single camera (default)
python3 main_dual_cam.py /dev/video0

# Optional: dual camera
python3 main_dual_cam.py /dev/video0 /dev/video1
```

**First Run After Engine Delete:** TensorRT SGIE engine rebuild takes ~5-8 minutes. Subsequent runs start in seconds.

---

## 🔍 Verifying Everything Works

### Expected Console Output (Healthy Pipeline)

```
[INFO] Loaded 3 enrolled face(s) from database
  → Sharad  dim=512  norm=1.0000
  → Aditya  dim=512  norm=1.0000
  → Raj     dim=512  norm=1.0000
[INFO] Watchlist matrix built: (3, 512)  (N=3, D=512)
============================================================
  JETSON NANO — Face Recognition Pipeline  (v2)
============================================================
  Sources         : 1
  Enrolled Faces  : 3
  Threshold       : 0.4
  Watchlist Matrix: READY (3, 512)
  Cache Eviction  : every 100 frames (stale > 150 frames)
============================================================
Creating Primary GIE (YOLOv8 Face)...
Creating Tracker...
Creating Secondary GIE (MobileFaceNet)...
...
Starting pipeline — press Ctrl+C to stop

[cam 0] Frame 30 | Faces: 1 | FPS: 22.1 | Cache: 1 IDs
  [SGIE] obj_id=1  dim=512  norm=1.0000  first3=[0.032 -0.015 0.048]
  [MATCH] obj_id=1  score=0.7823  thresh=0.40  → RECOGNISED
[ATTENDANCE] Sharad logged on cam 0
```

### Signs of Problems

| Symptom | Likely Cause | Fix |
|---|---|---|
| `pyds` import error | Running inside venv | `deactivate` and use system python3 |
| "No SGIE metadata" on all faces | Old SGIE engine with wrong batch-size | Delete `*.engine` in models/, restart |
| All faces show "Unknown" | Enrollment used different model | Re-enroll with same `w600k_mbf.onnx` |
| FPS < 5 | GPU overloaded | Increase PGIE `interval` to 6 |
| Segfault on multi-face | Old engine with batch-size=1 | Delete SGIE `.engine` file and restart |

---

## 🚀 Future Improvements (Next Steps)

### Priority 1: Robustness
- [ ] Anti-spoofing / liveness detection (blink detection or depth)
- [ ] Face alignment using landmarks before SGIE
- [ ] Re-identification when tracked ID is lost and re-appears

### Priority 2: Scalability
- [ ] Web dashboard (Flask/FastAPI reading from `attendance.db`)
- [ ] Multi-Jetson via MQTT/Redis for distributed attendance
- [ ] Cloud sync for attendance logs

### Priority 3: Performance
- [ ] INT8 quantization for PGIE/SGIE
- [ ] Adaptive PGIE interval based on current FPS
- [ ] Run SGIE only every N frames per tracked object

### Priority 4: UX
- [ ] Audio feedback (buzzer/speaker on successful attendance)
- [ ] GPIO LED indicator (green=recognised, red=unknown)
- [ ] Headless mode: replace `nveglglessink` with `fakesink` + MQTT

---

## 📋 Dependency Summary

### System-Level (via JetPack 4.6.1 — already installed)

| Package | Version |
|---|---|
| CUDA | 10.2 |
| TensorRT | 8.2 |
| DeepStream SDK | 6.0.1 |
| GStreamer | 1.14.5 |
| Python | 3.6.9 |
| pyds | 1.1.1 |
| numpy | system pip3 |

### Virtual Environment (for enrollment only)

| Package | Install Command |
|---|---|
| numpy | `pip install numpy` |
| opencv-python | `pip install opencv-python` |
| onnxruntime | `pip install onnxruntime` |

---

## ⚠️ Critical Reminders

1. **NEVER run `main_dual_cam.py` inside the venv** — `pyds` and GStreamer will not link
2. **Delete old SGIE `.engine`** after changing batch-size in config
3. **Enrollment and pipeline must use the SAME `w600k_mbf.onnx`**
4. **First TensorRT rebuild is slow** (~5-8 min) — don't kill the process
5. **Check camera index:** `v4l2-ctl --list-devices`
