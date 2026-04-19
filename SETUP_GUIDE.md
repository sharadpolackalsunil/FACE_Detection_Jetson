# Setup Guide

> **This file is superseded by [JETSON_SETUP.md](JETSON_SETUP.md).**
>
> Refer to `JETSON_SETUP.md` for the complete, up-to-date setup instructions,
> continuation checklist, and deployment guide for the Jetson Nano.

## Quick Reference

### Enroll Faces (in venv)
```bash
source venv/bin/activate
python3 enroll_trt.py sharad_front.jpg "Sharad_1"
python3 enroll_trt.py sharad_left.jpg "Sharad_2"
python3 enroll_trt.py sharad_right.jpg "Sharad_3"
python3 enroll_trt.py aditya.jpg "Aditya"
python3 enroll_trt.py RAJ.jpg "Raj"
deactivate
```

### Run Pipeline (SYSTEM Python — no venv!)
```bash
deactivate 2>/dev/null
python3 main_dual_cam.py /dev/video0
```

See [JETSON_SETUP.md](JETSON_SETUP.md) for the full walkthrough.
