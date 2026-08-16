"""
network/tcp_client.py
======================
Single shared TCP connection to the vehicle controller (port 4545).
Both the driving code (VehicleController / LaneKeepController) and the
mission code (MissionRunner) send commands through one TCPClient
instance so they never race on two separate sockets.
"""

import socket
import threading
import time

from config import PID_ENABLE_VALUE, GRIPPER_PID_ENABLE_VALUE

# ─── Network ────────────────────────────────────────────────────────────────

class TCPClient:
    """Shared TCP connection to the vehicle controller (port 4545).

    Used both by the manual/lane-keeping drive loop (VehicleController,
    fire-and-forget `send()`) and by the AprilTag-centering / gripper
    mission runner (MissionRunner), which additionally needs `query()` to
    read back encoder replies (e.g. "ENC3?" during gripper homing). There is
    only ever ONE socket to the vehicle - both call sites share this class -
    so driving commands and gripper/centering commands never race on two
    separate connections.
    """

    RECONNECT_COOLDOWN_SEC = 2.0   # min gap between auto-reconnect attempts

    def __init__(self):
        self.sock = None
        self.connected = False
        self.lock = threading.Lock()
        self._recv_buf = b""
        self._host = None
        self._port = None
        self._last_reconnect_attempt = 0.0

    def connect(self, host, port):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(3)
            self.sock.connect((host, port))
            self.connected = True
            self._recv_buf = b""
            self._host, self._port = host, port
            return True
        except Exception as e:
            self.connected = False
            return str(e)

    def disconnect(self):
        self.connected = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
        self._recv_buf = b""

    def _auto_reconnect_locked(self):
        """Best-effort silent reconnect using the last known host/port,
        rate limited. Must be called with self.lock held. Returns True if a
        usable socket is available afterwards."""
        if self.sock is not None:
            return True
        if not self._host:
            return False
        now = time.time()
        if now - self._last_reconnect_attempt < self.RECONNECT_COOLDOWN_SEC:
            return False
        self._last_reconnect_attempt = now
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3)
            s.connect((self._host, self._port))
            self.sock = s
            self.connected = True
            self._recv_buf = b""
            return True
        except Exception:
            self.sock = None
            self.connected = False
            return False

    def send(self, message):
        with self.lock:
            if self.sock is None and not self._auto_reconnect_locked():
                self.connected = False
                return False
            try:
                self.sock.sendall((message + "\n").encode("utf-8"))
                return True
            except Exception:
                self.connected = False
                self.sock = None
                self._recv_buf = b""
                return False

    def query(self, message, bufsize=128, timeout=1.0):
        """Send a command that expects a text reply (e.g. "ENC3?") and
        return the reply string, or None on failure/timeout. Blocks the
        calling thread for up to `timeout` seconds - call from a background
        thread, never from the Tk main loop. Buffers across multiple
        recv() calls and looks for a line terminator instead of assuming a
        single recv() contains exactly one full reply (TCP gives no such
        guarantee)."""
        with self.lock:
            if self.sock is None and not self._auto_reconnect_locked():
                self.connected = False
                return None
            try:
                self.sock.settimeout(timeout)
                self.sock.sendall((message + "\n").encode("utf-8"))

                deadline = time.time() + timeout
                while b"\n" not in self._recv_buf:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        break
                    self.sock.settimeout(remaining)
                    chunk = self.sock.recv(bufsize)
                    if not chunk:
                        raise ConnectionError("Socket closed by peer")
                    self._recv_buf += chunk

                if b"\n" in self._recv_buf:
                    line, self._recv_buf = self._recv_buf.split(b"\n", 1)
                else:
                    line, self._recv_buf = self._recv_buf, b""

                self.sock.settimeout(3)
                if not line:
                    return None
                return line.decode(errors="replace").strip()
            except socket.timeout:
                return None
            except Exception:
                self.connected = False
                self.sock = None
                self._recv_buf = b""
                return None

    # -- convenience wrappers used by MissionRunner (STATE 2 / STATE 3) -----
    def reset_encoders(self):
        self.send("RESET")

    def enable_drive_pid(self):
        self.send(f"PID1 {PID_ENABLE_VALUE}")
        self.send(f"PID2 {PID_ENABLE_VALUE}")

    def set_centering_position(self, deg1, deg2):
        self.send(f"SET1 {deg1}")
        self.send(f"SET2 {deg2}")

    def enable_gripper_pid(self):
        self.send(f"PID3 {GRIPPER_PID_ENABLE_VALUE}")

    def set_gripper_angle(self, deg):
        self.send(f"SET3 {deg}")

    def query_gripper_encoder(self):
        return self.query("ENC3?")

