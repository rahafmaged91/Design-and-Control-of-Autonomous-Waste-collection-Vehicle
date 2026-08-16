"""
vision/mjpeg_reader.py
========================
Background reader for the ESP32-CAM's MJPEG stream (TCP/HTTP port
8080). Runs continuously for the whole lifetime of the app so a fresh
frame is always available for the AprilTag-centering / stop-sign
watcher (STATE 1), regardless of what the mission state machine is
doing.
"""

import threading
import queue
import time
import urllib.request

import cv2
import numpy as np

from config import MJPEG_QUEUE_SIZE, MJPEG_CHUNK_SIZE

# ─── MJPEG stream reader (ESP32-CAM, port 8080) - runs continuously ─────────

class MJPEGReader:
    def __init__(self, ip, port, path="/video", log_cb=print):
        self._lock = threading.Lock()
        self.ip = ip
        self.port = port
        self.path = path
        self._log = log_cb
        self.q = queue.Queue(maxsize=MJPEG_QUEUE_SIZE)
        self.running = False
        self._force_reconnect = threading.Event()
        self.last_frame_size = None

    def start(self):
        if self.running:
            return
        self.running = True
        threading.Thread(target=self._loop, daemon=True, name="MJPEGReader").start()

    def reconnect(self, ip=None, port=None, path=None):
        with self._lock:
            if ip is not None:
                self.ip = ip
            if port is not None:
                self.port = port
            if path is not None:
                self.path = path
        self._log(f"[STREAM] Reconnect requested -> http://{self.ip}:{self.port}{self.path}")
        self._force_reconnect.set()

    def _endpoints(self):
        with self._lock:
            ip, port, path = self.ip, self.port, self.path
        primary = f"http://{ip}:{port}{path}"
        return [primary, f"http://{ip}:{port}/stream",
                f"http://{ip}:{port}/mjpeg", f"http://{ip}:{port}/"]

    def _loop(self):
        stream = None
        buf = b""
        endpoints = self._endpoints()
        ep_idx = 0
        backoff = 1.0

        while self.running:
            try:
                if self._force_reconnect.is_set():
                    self._force_reconnect.clear()
                    stream = None
                    endpoints = self._endpoints()
                    ep_idx = 0
                    backoff = 1.0

                if stream is None:
                    url = endpoints[ep_idx % len(endpoints)]
                    req = urllib.request.Request(url)
                    req.add_header("User-Agent", "Mozilla/5.0")
                    stream = urllib.request.urlopen(req, timeout=10)
                    buf = b""
                    self._log(f"[STREAM] Connected: {url}")
                    backoff = 1.0

                chunk = stream.read(MJPEG_CHUNK_SIZE)
                if not chunk:
                    raise ConnectionError("Empty read - stream closed")
                buf += chunk

                while True:
                    s = buf.find(b"\xff\xd8")
                    e = buf.find(b"\xff\xd9", s)
                    if s == -1 or e == -1:
                        break
                    jpg = buf[s:e + 2]
                    buf = buf[e + 2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    self.last_frame_size = (frame.shape[1], frame.shape[0])
                    if self.q.full():
                        try:
                            self.q.get_nowait()
                        except queue.Empty:
                            pass
                    self.q.put(frame)

                if len(buf) > 500_000:
                    buf = buf[-100_000:]

            except Exception as e:
                self._log(f"[STREAM] Error: {e} - retrying in {backoff:.1f}s")
                stream = None
                ep_idx += 1
                time.sleep(backoff)
                backoff = min(backoff * 1.7, 8.0)

    def read_frame(self, timeout=2):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False


