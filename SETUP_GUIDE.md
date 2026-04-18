# Setup Guide

> **This file is superseded by [JETSON_SETUP.md](JETSON_SETUP.md).**
>
> Refer to `JETSON_SETUP.md` for the complete, up-to-date setup instructions,
> continuation checklist, and deployment guide for the Jetson Nano.

## Quick Reference

### Enroll Faces (in venv)
```bash
source venv/bin/activate
python3 enroll_trt.py face_image.jpg "Person Name"
deactivate
```

### Run Pipeline (SYSTEM Python — no venv!)
```bash
deactivate 2>/dev/null
python3 main_dual_cam.py /dev/video0
```

See [JETSON_SETUP.md](JETSON_SETUP.md) for the full walkthrough.
