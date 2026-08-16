"""
camera_utils.py
================
Small helpers for finding and opening a local webcam (used by the PC
camera side of the app, not the ESP32-CAM MJPEG stream).
"""

import sys

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ─── Camera helpers ──────────────────────────────────────────────────────────

def scan_cameras(max_index=6, log_cb=print):
    """Try camera indices 0..max_index-1 (DirectShow fallback on Windows,
    needed for virtual cameras like DroidCam) and report working ones."""
    if not CV2_AVAILABLE:
        log_cb("Cannot scan cameras: opencv-python/numpy not installed.")
        return []

    backends = [("default", None)]
    if sys.platform.startswith("win"):
        backends.append(("DSHOW", cv2.CAP_DSHOW))

    working = []
    for idx in range(max_index):
        for backend_name, backend_flag in backends:
            try:
                cap = (cv2.VideoCapture(idx) if backend_flag is None
                       else cv2.VideoCapture(idx, backend_flag))
            except Exception as e:
                log_cb(f"  index {idx} [{backend_name}]: error opening ({e})")
                continue
            if not cap.isOpened():
                cap.release()
                continue
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                log_cb(f"  index {idx} [{backend_name}]: OK, frame {w}x{h}")
                working.append({"index": idx, "backend": backend_name,
                                "backend_flag": backend_flag,
                                "width": w, "height": h})
            else:
                log_cb(f"  index {idx} [{backend_name}]: opened but no frame")
            cap.release()

    if not working:
        log_cb("No camera indices responded. Check the camera source is "
               "running, not in use elsewhere, and permissions are granted.")
    return working


def open_camera(index, log_cb=print):
    """Open a camera index, falling back to DirectShow on Windows."""
    if not CV2_AVAILABLE:
        return None
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap
        cap.release()
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                log_cb(f"Opened camera {index} using DirectShow backend.")
                return cap
            cap.release()
    return None

