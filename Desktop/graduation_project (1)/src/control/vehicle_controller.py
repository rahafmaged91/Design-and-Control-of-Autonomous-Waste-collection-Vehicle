"""
control/vehicle_controller.py
===============================
Top-level driving host: manual W/A/S/D control loop with speed/steer
ramping, path record + playback, and ownership of the LaneKeepController
(autonomous STATE 1). This is what the GUI's key bindings and buttons
talk to.
"""

import os
import threading
import time

from config import (
    SPEED_MAX_VALUE, STEER_CENTER, STEER_MAX_RIGHT, STEER_MAX_LEFT,
)
from network.tcp_client import TCPClient
from control.lane_keep_controller import LaneKeepController

# ─── Vehicle controller (manual drive + record/playback + autonomy host) ────

class VehicleController:
    SPEED_MAX = SPEED_MAX_VALUE
    SPEED_STEP = 40          # per tick while key held
    SPEED_DECAY = 30         # per tick when released
    STEER_STEP = 160         # == mechanical slew limit per 50 ms tick
    STEER_DECAY = 120
    TICK_MS = 50             # update interval

    def __init__(self, tcp: TCPClient, status_cb, log_cb):
        self.tcp = tcp
        self.status_cb = status_cb
        self.log_cb = log_cb

        self.steer_center = STEER_CENTER
        self.steer_min = STEER_MAX_RIGHT
        self.steer_max = STEER_MAX_LEFT

        self.speed = 0
        self.steer = self.steer_center

        self.keys = {"w": False, "s": False, "a": False, "d": False}

        self._running = False
        self._thread = None

        self.recording = False
        self.record_start = None
        self.recorded_path = []
        self.playing = False
        self._playback_thread = None
        self.active_path = []
        self.active_source = None

        self.autonomous = False
        self._lane_keeper = None

        self.mission_active = False

    def set_mission_active(self, flag):
        self.mission_active = bool(flag)

    def key_down(self, k):
        if k in self.keys:
            self.keys[k] = True

    def key_up(self, k):
        if k in self.keys:
            self.keys[k] = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self.stop_autonomous()

    def _loop(self):
        while self._running:
            self._update()
            time.sleep(self.TICK_MS / 1000)

    def _update(self):
        if self.playing or self.autonomous or self.mission_active:
            return

        if self.keys["w"] and not self.keys["s"]:
            self.speed = min(self.speed + self.SPEED_STEP, self.SPEED_MAX)
        elif self.keys["s"] and not self.keys["w"]:
            self.speed = max(self.speed - self.SPEED_STEP, -self.SPEED_MAX)
        else:
            if self.speed > 0:
                self.speed = max(0, self.speed - self.SPEED_DECAY)
            elif self.speed < 0:
                self.speed = min(0, self.speed + self.SPEED_DECAY)

        if self.keys["d"] and not self.keys["a"]:
            self.steer = max(self.steer - self.STEER_STEP, self.steer_min)
        elif self.keys["a"] and not self.keys["d"]:
            self.steer = min(self.steer + self.STEER_STEP, self.steer_max)
        else:
            if self.steer > self.steer_center:
                self.steer = max(self.steer_center, self.steer - self.STEER_DECAY)
            elif self.steer < self.steer_center:
                self.steer = min(self.steer_center, self.steer + self.STEER_DECAY)

        self._send_commands()

        if self.recording:
            elapsed_ms = int((time.time() - self.record_start) * 1000)
            self.recorded_path.append((elapsed_ms, self.speed, self.steer))

    def _send_commands(self):
        if not self.tcp.connected:
            self.status_cb(self.speed, self.steer)
            return
        spd = abs(self.speed)
        if self.speed >= 0:
            cmd1 = f"SET1 {spd} 0"
            cmd2 = f"SET2 {spd} 0"
        else:
            cmd1 = f"SET1 0 {spd}"
            cmd2 = f"SET2 0 {spd}"
        self.tcp.send(cmd1)
        self.tcp.send(cmd2)
        self.tcp.send(f"SERVO1 SET {self.steer}")
        self.status_cb(self.speed, self.steer)

    def send_stop_command(self):
        self.tcp.send("STOP")

    def send_servo_config(self):
        self.tcp.send(f"SERVO1 MIN {self.steer_min}")
        self.tcp.send(f"SERVO1 MAX {self.steer_max}")
        self.log_cb(f"Sent: SERVO1 MIN {self.steer_min}  |  "
                    f"SERVO1 MAX {self.steer_max}")

    def toggle_recording(self):
        if self.playing:
            self.log_cb("Can't record while a path is playing back.")
            return
        if self.autonomous:
            self.log_cb("Can't record while lane tracking is active.")
            return
        if self.mission_active:
            self.log_cb("Can't record while the AprilTag/gripper mission is active.")
            return
        if self.recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        self.recording = True
        self.record_start = time.time()
        self.recorded_path = []
        self.log_cb("Recording path... press R again to stop and save.")

    def _stop_recording(self):
        self.recording = False
        if not self.recorded_path:
            self.log_cb("No path recorded (nothing captured).")
            return
        filename = f"path_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        try:
            with open(filename, "w") as f:
                for elapsed_ms, speed, steer in self.recorded_path:
                    f.write(f"{elapsed_ms},{speed},{steer}\n")
            self.active_path = list(self.recorded_path)
            self.active_source = filename
            self.log_cb(f"Recording stopped. Saved {len(self.recorded_path)} "
                        f"points to {filename}")
        except Exception as e:
            self.log_cb(f"Failed to save path: {e}")

    def load_path_file(self, filename):
        if not os.path.exists(filename):
            self.log_cb(f"Path file not found: {filename}")
            return False
        pts = self._load_path_file(filename)
        if not pts:
            self.log_cb(f"Path file is empty or invalid: {filename}")
            return False
        self.active_path = pts
        self.active_source = filename
        self.log_cb(f"Loaded path from {os.path.basename(filename)} "
                    f"({len(pts)} points). Press P to play it.")
        return True

    def play_path(self):
        if self.recording:
            self.log_cb("Stop recording (press R) before playing back a path.")
            return
        if self.autonomous:
            self.log_cb("Stop lane tracking (press L) before playback.")
            return
        if self.mission_active:
            self.log_cb("Can't play back a path while the AprilTag/gripper mission is active.")
            return
        if self.playing:
            self.log_cb("Already playing a path.")
            return
        if not self.active_path:
            self.log_cb("No path loaded. Press R to record one, or LOAD PATH FILE.")
            return
        path = self.active_path
        source = self.active_source or "recording"
        self.log_cb(f"Playing back path from {source} ({len(path)} points)...")
        self._playback_thread = threading.Thread(
            target=self._playback_loop, args=(path,), daemon=True)
        self._playback_thread.start()

    def _load_path_file(self, filename):
        pts = []
        try:
            with open(filename) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    elapsed_ms, speed, steer = line.split(",")
                    pts.append((int(elapsed_ms), int(speed), int(steer)))
        except Exception as e:
            self.log_cb(f"Failed to load path file: {e}")
            return []
        return pts

    def _playback_loop(self, path):
        self.playing = True
        prev_t = 0
        try:
            for elapsed_ms, speed, steer in path:
                if not self.playing:
                    break
                time.sleep(max(0.0, (elapsed_ms - prev_t) / 1000))
                prev_t = elapsed_ms
                self.speed = speed
                self.steer = steer
                self._send_commands()
        finally:
            was_stopped_early = not self.playing
            self.playing = False
            self.log_cb("Playback stopped." if was_stopped_early
                        else "Playback finished.")

    def stop_playback(self):
        self.playing = False

    def start_autonomous(self, camera_index=0, frame_cb=None,
                         confirm_cb=None, params_cb=None, road_type=None):
        if self.recording:
            self.log_cb("Stop recording (R) before starting lane tracking.")
            return False
        if self.playing:
            self.log_cb("Stop playback before starting lane tracking.")
            return False
        if self.autonomous:
            self.log_cb("Lane tracking already running.")
            return False
        if self.mission_active:
            self.log_cb("Can't start lane tracking while the AprilTag/gripper mission is active.")
            return False
        self._lane_keeper = LaneKeepController(
            self, camera_index=camera_index, log_cb=self.log_cb,
            frame_cb=frame_cb, confirm_cb=confirm_cb, params_cb=params_cb,
            road_type=road_type)
        ok = self._lane_keeper.start()
        if ok:
            self.autonomous = True
        else:
            self._lane_keeper = None
        return ok

    def stop_autonomous(self):
        if not self.autonomous:
            return
        self.autonomous = False
        if self._lane_keeper:
            self._lane_keeper.stop()
            self._lane_keeper = None

    def confirm_lane(self, shift=0):
        if self._lane_keeper:
            self._lane_keeper.confirm_lane(shift)

    def request_lane_change(self, direction):
        if self._lane_keeper:
            return self._lane_keeper.model.request_lane_change(direction)
        return False


