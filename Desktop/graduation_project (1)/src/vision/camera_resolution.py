"""
vision/camera_resolution.py
=============================
Sends a UDP resolution-change request to the ESP32-CAM so the stream
resolution can be adjusted live from the GUI.
"""

import socket

from config import FRAMESIZE_PRESETS, RESOLUTION_UDP_PORT

def set_camera_resolution(ip, port, preset_name, log_cb=print):
    """Send a resolution-change command to the ESP32 over UDP (numeric
    preset code). This is the "adjustable stream resolution" feature."""
    val = FRAMESIZE_PRESETS.get(preset_name)
    if val is None:
        log_cb(f"[RES] Unknown preset: {preset_name}")
        return False
    cmd = str(val)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(3.0)
        sock.sendto(cmd.encode("ascii"), (ip, RESOLUTION_UDP_PORT))
        try:
            reply, _ = sock.recvfrom(256)
            reply_txt = reply.decode(errors="replace").strip()
        except socket.timeout:
            reply_txt = "(no reply)"
        sock.close()
        log_cb(f"[RES] Sent UDP '{cmd}' -> {ip}:{RESOLUTION_UDP_PORT} "
               f"({preset_name}) | ESP32 reply: {reply_txt}")
        return True
    except Exception as e:
        log_cb(f"[RES] UDP resolution request failed ({ip}:{RESOLUTION_UDP_PORT}): {e}. "
               "The stream will still auto-adapt to whatever size frames actually arrive at.")
        return False


