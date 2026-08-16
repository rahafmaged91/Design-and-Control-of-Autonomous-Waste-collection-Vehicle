"""
mission/mission_helpers.py
============================
Small standalone helpers used by the mission state machine: parsing an
integer out of a controller reply string, converting a linear distance
to a wheel-rotation angle, and the (not-yet-implemented) stop-sign
detector hook for STATE 2b.
"""

import re

from config import WHEEL_CIRCUMFERENCE_MM

def _parse_int(text):
    """Pull the first signed integer out of a controller reply like
    'ENC3 10' or '10'."""
    if not text:
        return None
    m = re.search(r'-?\d+', text)
    if not m:
        return None
    try:
        return int(m.group())
    except ValueError:
        return None


def mm_to_wheel_degrees(distance_mm):
    """Convert a linear distance (mm) to the wheel-rotation angle (degrees)
    needed to travel it, given WHEEL_DIAMETER_MM."""
    return (distance_mm / WHEEL_CIRCUMFERENCE_MM) * 360.0


def detect_stop_sign(frame_bgr):
    """STOP-SIGN DETECTION ON THE PC CAMERA STREAM - STILL TO BE IMPLEMENTED.

    Placeholder hook for STATE 2b's other trigger condition. Wire in a real
    detector here (e.g. a trained classifier, or a red-octagon color/shape
    heuristic) and have it return True once a stop sign is confidently seen
    in `frame_bgr`. Until then this always reports "not detected", so
    STATE 2 can only ever be reached via the AprilTag branch.
    """
    return False


