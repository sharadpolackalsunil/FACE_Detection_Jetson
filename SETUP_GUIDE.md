# Setup Guide

> **This file is superseded by [JETSON_SETUP.md](JETSON_SETUP.md).**
>
> Refer to `JETSON_SETUP.md` for the complete, up-to-date setup instructions,
> continuation checklist, and deployment guide for the Jetson Nano.

## Quick Reference

### 1. Prepare Face Images (local directory)
```
image_db/
├── sharad/
│   ├── sharad_1.jpg
│   ├── sharad_2.jpg
│   └── sharad_3.jpg
├── aditya/
│   ├── aditya_1.jpg
│   └── aditya_2.jpg
└── raj/
    ├── raj_1.jpg
    └── raj_2.jpg
```
Place 3-4 photos per person (different angles for best accuracy).

### 2. Enroll Faces (in venv — auto-scan mode)
```bash
source venv/bin/activate
<<<<<<< HEAD
python3 enroll_trt.py sharad_front.jpg "Sharad_1"
python3 enroll_trt.py sharad_left.jpg "Sharad_2"
python3 enroll_trt.py sharad_right.jpg "Sharad_3"
python3 enroll_trt.py aditya.jpg "Aditya"
python3 enroll_trt.py RAJ.jpg "Raj"
=======
python3 enroll_trt.py
>>>>>>> sharad
deactivate
```
This automatically scans `image_db/`, skips students already enrolled, and averages embeddings from all images per student.

### 3. Run Pipeline (SYSTEM Python — no venv!)
```bash
deactivate 2>/dev/null
python3 main_dual_cam.py /dev/video0
```

### 4. View Attendance
```bash
cat attendance.csv
```
Columns: `Name`, `Date`, `Time` — auto-deduped per 5-minute window.

See [JETSON_SETUP.md](JETSON_SETUP.md) for the full walkthrough.
