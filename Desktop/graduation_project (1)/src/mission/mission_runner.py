"""
mission/mission_runner.py
===========================
The full mission state machine: watches the ESP32-CAM stream during
STATE 1 (LANE_FOLLOWING) for an AprilTag or a stop sign, drives the
STATE 2 transition/centering/stop-sign-halt states, runs the STATE 3
gripper sequence, and handles the AWAITING_RESET hand-off back to the
user. See the README's "State Machine Description" section for the
full diagram.
"""

import queue
import threading
import time

import cv2

from config import (
    ESP32_CAM_IP, ESP32_CAM_PORT, ESP32_STREAM_PATH,
    TAG_CALIB_WIDTH, TAG_CALIB_HEIGHT, TAG_CONFIRM_FRAMES,
    TAG_TRANSITION_DELAY_SEC, TAG_CENTER_CMD_INTERVAL,
    TAG_CENTER_LOCK_FRAMES, TAG_CENTER_LOCK_MISS_PENALTY,
    TAG_ERROR_MARGIN_MM, TAG_PID_KP, TAG_PID_KI, TAG_PID_KD,
    TAG_PID_OUT_MIN_MM, TAG_PID_OUT_MAX_MM, TAG_MAX_CMD_DEGREES,
    TAG_MIN_CMD_DELTA_DEG, TAG_LOST_HOLD_TIMEOUT, TAG_DETECTION_SCALE,
    TAG_MIN_MARGIN, UNDISTORT_TAG_FRAME, STOPSIGN_HALT_SEC,
    PID_ENABLE_VALUE, GRIPPER_PID_ENABLE_VALUE, GRIPPER_ABORT_POLL_SEC,
    GRIPPER_HOMING_STEP_DEG, GRIPPER_HOMING_STEP_INTERVAL,
    GRIPPER_HOMING_POLL_INTERVAL, GRIPPER_HOMING_STABLE_TOL,
    GRIPPER_HOMING_STABLE_COUNT, GRIPPER_HOMING_MAX_ANGLE_MAG,
    GRIPPER_HOMING_TIMEOUT_SEC, GRIPPER_RAMP_MODE, GRIPPER_RAMP_MAX_ANGLE,
    GRIPPER_RAMP_STEP, GRIPPER_RAMP_INTERVAL, GRIPPER_HOLD_AT_MAX_SEC,
    GRIPPER_POST_WAIT_SEC,
)
from mission.mission_enums import MissionState, GripperPhase, GripperAborted
from mission.mission_helpers import _parse_int, mm_to_wheel_degrees, detect_stop_sign
from vision.mjpeg_reader import MJPEGReader
from vision.apriltag_utils import (
    build_base_tag_camera_matrix, scale_tag_camera_matrix,
    build_tag_undistort_maps, run_tag_detector_multipass,
    get_tag_pose, get_tag_yaw, TagObservation, TAG_DIST_COEFFS,
)
from control.center_pid import CenterPIDController
from network.tcp_client import TCPClient

try:
    from pupil_apriltags import Detector as AprilTagDetector
    APRILTAG_AVAILABLE = True
except ImportError:
    APRILTAG_AVAILABLE = False

# ─── MissionRunner: STATE 1 watcher + STATE 2 (centering) + STATE 3 (gripper) ─

class MissionRunner:
    """Owns the 8080 MJPEG stream + AprilTag detector and drives the
    STATE 2 / STATE 3 half of the overall system:

      LANE_FOLLOWING -> (tag confirmed) -> TAG_TRANSITION -> CENTERING
                     -> (locked, stable) -> GRIPPER -> AWAITING_RESET
                     -> (user confirms)  -> "RESET" sent -> LANE_FOLLOWING

      LANE_FOLLOWING -> (stop sign, stub) -> STOPSIGN_HALT
                     -> (5s elapsed)       -> LANE_FOLLOWING

    Runs its own background thread reading the 8080 stream continuously
    (so the preview + tag watch work regardless of what STATE 1's lane
    camera is doing), and shares the single vehicle TCPClient connection
    (port 4545) with VehicleController so drive / centering / gripper
    commands never race on two separate sockets.
    """

    def __init__(self, tcp: TCPClient, vehicle_ctrl, log_cb,
                 restart_lane_follow_cb, esp_ip=ESP32_CAM_IP,
                 esp_port=ESP32_CAM_PORT, esp_path=ESP32_STREAM_PATH):
        self.tcp = tcp
        self.vc = vehicle_ctrl
        self.log = log_cb
        self._restart_lane_follow_cb = restart_lane_follow_cb

        self.reader = MJPEGReader(esp_ip, esp_port, esp_path, log_cb=log_cb)
        self.frame_q = queue.Queue(maxsize=4)   # for the GUI preview

        self.detector = None
        if APRILTAG_AVAILABLE:
            self.detector = AprilTagDetector(
                families="tag36h11", nthreads=4, quad_decimate=1.0,
                quad_sigma=0.8, refine_edges=1, decode_sharpening=0.25)

        self._base_K = build_base_tag_camera_matrix()
        self._active_K = None
        self._map1 = self._map2 = None
        self._current_frame_size = None

        self.state = MissionState.IDLE
        self.state_entered_at = time.time()
        self._stop = False
        self._estop = threading.Event()

        # STATE 1 watcher bookkeeping
        self._tag_hits = 0

        # STATE 2a centering bookkeeping
        self.pid = CenterPIDController(TAG_PID_KP, TAG_PID_KI, TAG_PID_KD,
                                       TAG_PID_OUT_MIN_MM, TAG_PID_OUT_MAX_MM,
                                       deadband=TAG_ERROR_MARGIN_MM)
        self._last_cmd_time = 0.0
        self._last_cmd_deg = None
        self._lost_tag_since = None
        self._center_lock_count = 0
        self._centering_holding = False   # True while STOP-holding on a lost tag

        # STATE 3 gripper bookkeeping (tunable via update_gripper_params())
        self.gripper_ramp_step = GRIPPER_RAMP_STEP
        self.gripper_ramp_interval = GRIPPER_RAMP_INTERVAL
        self.gripper_hold_sec = GRIPPER_HOLD_AT_MAX_SEC
        self._gripper_thread = None
        self._gripper_abort = threading.Event()
        self.gripper_phase = None
        self.gripper_angle = 0
        self.gripper_last_enc = None
        self.gripper_stable_count = 0

        # published debug/telemetry fields (read by the GUI)
        self.tag_id = None
        self.last_margin = 0.0
        self.distance_cm = 0.0
        self.offset_cm = 0.0
        self.yaw_deg = 0.0
        self.pixel_error = 0.0
        self.pid_output_mm = 0.0
        self.cmd_degrees = 0
        self.trigger = None            # "apriltag" | "stopsign" | None

        self._dispatch = {
            MissionState.LANE_FOLLOWING: self._tick_lane_following,
            MissionState.TAG_TRANSITION: self._tick_tag_transition,
            MissionState.CENTERING:      self._tick_centering,
            MissionState.STOPSIGN_HALT:  self._tick_stopsign_halt,
        }

    # ------------------------------------------------------------------ #
    def start(self):
        self.reader.start()
        threading.Thread(target=self._run, daemon=True, name="MissionRunner").start()

    def halt(self):
        self._stop = True
        self._gripper_abort.set()
        self.reader.stop()

    def _set_state(self, new_state):
        self.log(f"[MISSION] {self.state.name} -> {new_state.name}")
        self.state = new_state
        self.state_entered_at = time.time()

    # ------------------------------------------------------------------ #
    def begin_lane_following(self):
        self._estop.clear()
        self._tag_hits = 0
        self.trigger = None
        self._set_state(MissionState.LANE_FOLLOWING)

    def lane_following_stopped_by_user(self):
        if self.state == MissionState.LANE_FOLLOWING:
            self._set_state(MissionState.IDLE)

    # ------------------------------------------------------------------ #
    def update_centering_params(self, kp=None, ki=None, kd=None, error_margin_mm=None):
        self.pid.set_gains(kp=kp, ki=ki, kd=kd)
        if error_margin_mm is not None:
            self.pid.set_deadband(error_margin_mm)
        self.log(f"[MISSION] Centering PID updated -> Kp={self.pid.kp:.4f} "
                 f"Ki={self.pid.ki:.4f} Kd={self.pid.kd:.4f} "
                 f"ErrorMargin={self.pid.deadband:.2f}mm")

    def update_gripper_params(self, step_deg=None, step_time_sec=None, pause_sec=None):
        if step_deg is not None:
            self.gripper_ramp_step = max(1, int(step_deg))
        if step_time_sec is not None:
            self.gripper_ramp_interval = max(0.01, float(step_time_sec))
        if pause_sec is not None:
            self.gripper_hold_sec = max(0.0, float(pause_sec))
        self.log(f"[MISSION] Gripper params updated -> step={self.gripper_ramp_step}deg "
                 f"step_time={self.gripper_ramp_interval}s pause={self.gripper_hold_sec}s")

    # ------------------------------------------------------------------ #
    def emergency_stop(self):
        self._estop.set()
        self._abort_running_gripper_thread(join_timeout=1.0)
        self.vc.stop_autonomous()
        self.vc.send_stop_command()
        self.vc.set_mission_active(False)
        self.log("[MISSION] EMERGENCY STOP - motors halted, mission ticking paused")

    def is_estopped(self):
        return self._estop.is_set()

    def manual_reset_and_repeat(self):
        self._estop.clear()
        self._abort_running_gripper_thread()
        self.tcp.send("RESET")
        self.log("[MISSION] RESET sent to vehicle controller (port 4545) - "
                 "restarting the cycle")
        self.vc.set_mission_active(False)
        self._tag_hits = 0
        self.trigger = None
        self.gripper_phase = None
        if self._restart_lane_follow_cb:
            self._restart_lane_follow_cb()
        self._set_state(MissionState.LANE_FOLLOWING)

    # ------------------------------------------------------------------ #
    def on_lane_frame(self, frame_bgr):
        if self.state != MissionState.LANE_FOLLOWING or self._estop.is_set():
            return
        try:
            if detect_stop_sign(frame_bgr):
                self._enter_stopsign_halt()
        except Exception as e:
            self.log(f"[MISSION] Stop-sign check raised an exception: {e}")

    # ------------------------------------------------------------------ #
    def _ensure_intrinsics_for(self, w, h):
        if self._current_frame_size == (w, h) and self._active_K is not None:
            return
        self._active_K = scale_tag_camera_matrix(self._base_K, (TAG_CALIB_WIDTH, TAG_CALIB_HEIGHT), (w, h))
        if UNDISTORT_TAG_FRAME:
            try:
                self._map1, self._map2, self._active_K = build_tag_undistort_maps(
                    self._active_K, TAG_DIST_COEFFS, (w, h))
            except Exception as e:
                self.log(f"[MISSION] Undistort map build failed ({e}); continuing without undistortion")
                self._map1 = self._map2 = None
        self._current_frame_size = (w, h)

    def _best_observation(self, gray):
        if self.detector is None:
            return None
        try:
            results = run_tag_detector_multipass(self.detector, gray)
        except Exception:
            self.log("[MISSION] Detection pass raised an exception; skipping frame")
            return None
        best = None
        for r in results:
            if r.decision_margin < TAG_MIN_MARGIN:
                continue
            if best is None or r.decision_margin > best.decision_margin:
                best = r
        if best is None:
            return None
        obs = TagObservation()
        obs.tag_id = best.tag_id
        obs.margin = best.decision_margin
        sc = [(x / TAG_DETECTION_SCALE, y / TAG_DETECTION_SCALE) for x, y in best.corners]
        pts = [(int(x), int(y)) for x, y in sc]
        obs.corners_px = pts
        cx = sum(p[0] for p in pts) / 4.0
        cy = sum(p[1] for p in pts) / 4.0
        obs.center_px = (cx, cy)
        try:
            ok, rvec, tvec = get_tag_pose([[p[0], p[1]] for p in pts], self._active_K)
        except Exception as e:
            self.log(f"[MISSION] Pose estimation failed: {e}")
            ok = False
        if ok:
            obs.pose_ok = True
            obs.distance_mm = float(tvec[2]) * 1000.0
            obs.yaw_deg = get_tag_yaw(rvec)
        return obs

    def _draw_overlay(self, frame, obs):
        if obs is None:
            return
        pts = obs.corners_px
        for i in range(4):
            cv2.line(frame, pts[i], pts[(i + 1) % 4], (0, 255, 0), 2)
        cx, cy = int(obs.center_px[0]), int(obs.center_px[1])
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(frame, f"ID:{obs.tag_id}", (pts[0][0], pts[0][1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def _draw_hud(self, frame):
        h, w = frame.shape[:2]
        cv2.line(frame, (w // 2, 0), (w // 2, h), (60, 60, 60), 1)
        cv2.line(frame, (0, h // 2), (w, h // 2), (60, 60, 60), 1)
        cv2.rectangle(frame, (0, 0), (280, 20), (20, 20, 20), -1)
        state_txt = self.state.name
        if self.state is MissionState.GRIPPER and self.gripper_phase is not None:
            state_txt += f" / {self.gripper_phase.name}"
        cv2.putText(frame, f"MISSION: {state_txt}", (6, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # ------------------------------------------------------------------ #
    def _tick_lane_following(self, frame, obs):
        tag_found = obs is not None
        self._tag_hits = self._tag_hits + 1 if tag_found else 0
        if tag_found:
            self.tag_id = obs.tag_id
            self.last_margin = obs.margin
            if obs.pose_ok:
                self.distance_cm = obs.distance_mm / 10.0
                self.yaw_deg = obs.yaw_deg
                self.offset_cm = ((obs.center_px[0] - frame.shape[1] / 2.0)
                                  / self._active_K[0, 0] * obs.distance_mm) / 10.0
        if self._tag_hits >= TAG_CONFIRM_FRAMES:
            self._enter_tag_transition()

    def _enter_tag_transition(self):
        self.trigger = "apriltag"
        self.vc.stop_autonomous()
        self.vc.set_mission_active(True)
        self.vc.send_stop_command()
        self.log(f"[MISSION] AprilTag confirmed over {TAG_CONFIRM_FRAMES} frames "
                 "-> STOP sent, starting transition delay")
        self._set_state(MissionState.TAG_TRANSITION)

    def _enter_stopsign_halt(self):
        self.trigger = "stopsign"
        self.vc.stop_autonomous()
        self.vc.set_mission_active(True)
        self.vc.send_stop_command()
        self.log(f"[MISSION] Stop sign detected -> STOP sent, halting for "
                 f"{STOPSIGN_HALT_SEC:.0f}s")
        self._set_state(MissionState.STOPSIGN_HALT)

    # ------------------------------------------------------------------ #
    def _tick_tag_transition(self, frame, obs):
        remaining = TAG_TRANSITION_DELAY_SEC - (time.time() - self.state_entered_at)
        if remaining <= 0.0:
            self._enter_centering()

    def _enter_centering(self):
        self.tcp.reset_encoders()
        self.tcp.enable_drive_pid()
        self.pid.reset()
        self._last_cmd_time = 0.0
        self._last_cmd_deg = None
        self._lost_tag_since = None
        self._center_lock_count = 0
        self._centering_holding = False
        self._set_state(MissionState.CENTERING)
        self.log("[MISSION] Encoder reset + drive PID enabled -> centering starting")

    # ------------------------------------------------------------------ #
    def _tick_centering(self, frame, obs):
        """STATE 2a. IMPORTANT: SET1/SET2 here are ABSOLUTE POSITION
        targets (wheel degrees relative to the encoders zeroed by RESET
        in _enter_centering()), not velocity commands - exactly like
        SET3/ENC3 for the gripper. That means "commanding 0 degrees" does
        NOT mean "hold current position" - it means "drive back to the
        zeroed origin". See the PATCH NOTES at the top of this file.

        FIX: once the offset is inside the PID deadband, we must NOT
        send a fresh 0-degree position command (that would unwind
        whatever corrective rotation just centered the vehicle, which is
        exactly the "moves away instead of settling" bug). Instead we
        simply stop sending new position commands and let the wheels
        hold wherever they already are.
        """
        now = time.time()
        if obs is None or not obs.pose_ok:
            if self._lost_tag_since is None:
                self._lost_tag_since = now
                self.log("[MISSION] Tag lost during centering - holding")
            elif now - self._lost_tag_since >= TAG_LOST_HOLD_TIMEOUT and not self._centering_holding:
                self.tcp.send("STOP")
                self._centering_holding = True
                self.pid.reset()
                self._center_lock_count = 0
                self.log("[MISSION] Tag lost > "
                         f"{TAG_LOST_HOLD_TIMEOUT:.1f}s - holding in place (STOP)")
            return

        if self._lost_tag_since is not None or self._centering_holding:
            self._last_cmd_time = 0.0
            self._last_cmd_deg = None
            self._centering_holding = False
        self._lost_tag_since = None

        self.tag_id = obs.tag_id
        self.last_margin = obs.margin
        self.distance_cm = obs.distance_mm / 10.0
        self.yaw_deg = obs.yaw_deg

        frame_center_x = frame.shape[1] / 2.0
        pixel_error = obs.center_px[0] - frame_center_x
        focal_x = self._active_K[0, 0]
        offset_mm = (pixel_error / focal_x) * obs.distance_mm
        self.pixel_error = pixel_error
        self.offset_cm = offset_mm / 10.0

        output_mm = self.pid.update(offset_mm, now)
        self.pid_output_mm = output_mm

        in_margin = abs(offset_mm) <= self.pid.deadband

        if in_margin:
            # Inside the deadband: the PID output is (correctly) 0.0, but
            # under position-mode drive control that does NOT mean "stay
            # here" - it means "return to the zeroed encoder origin",
            # which would undo the very correction that just centered the
            # vehicle. So: send NOTHING and just hold the last commanded
            # position. cmd_degrees is reported as the last real command
            # (or 0 if none was ever sent) purely for telemetry.
            self.cmd_degrees = self._last_cmd_deg if self._last_cmd_deg is not None else 0
        else:
            degrees = mm_to_wheel_degrees(output_mm)
            degrees = max(-TAG_MAX_CMD_DEGREES, min(TAG_MAX_CMD_DEGREES, degrees))
            degrees_int = int(round(degrees))
            self.cmd_degrees = degrees_int

            send_due = (now - self._last_cmd_time) >= TAG_CENTER_CMD_INTERVAL
            changed_enough = (self._last_cmd_deg is None or
                              abs(degrees_int - self._last_cmd_deg) >= TAG_MIN_CMD_DELTA_DEG)
            if send_due and changed_enough:
                self.tcp.set_centering_position(degrees_int, degrees_int)
                self._last_cmd_time = now
                self._last_cmd_deg = degrees_int

        if in_margin:
            self._center_lock_count = min(self._center_lock_count + 1, TAG_CENTER_LOCK_FRAMES)
        else:
            self._center_lock_count = max(0, self._center_lock_count - TAG_CENTER_LOCK_MISS_PENALTY)

        if self._center_lock_count >= TAG_CENTER_LOCK_FRAMES:
            self.log("[MISSION] Tag centered and stable -> starting gripper sequence")
            self._enter_gripper()

    # ------------------------------------------------------------------ #
    def _tick_stopsign_halt(self, frame, obs):
        remaining = STOPSIGN_HALT_SEC - (time.time() - self.state_entered_at)
        if remaining <= 0.0:
            self.log("[MISSION] Stop-sign halt complete -> resuming lane following")
            self.vc.set_mission_active(False)
            self.trigger = None
            self._tag_hits = 0
            if self._restart_lane_follow_cb:
                self._restart_lane_follow_cb()
            self._set_state(MissionState.LANE_FOLLOWING)

    # ------------------------------------------------------------------ #
    def _abort_running_gripper_thread(self, join_timeout=3.0):
        t = self._gripper_thread
        if t is not None and t.is_alive():
            self.log("[MISSION] Stopping previous gripper sequence...")
            self._gripper_abort.set()
            t.join(timeout=join_timeout)
            if t.is_alive():
                self.log("[MISSION] WARNING: previous gripper thread did not exit in time")
        self._gripper_abort.clear()

    def _abortable_sleep(self, seconds):
        end = time.time() + seconds
        while True:
            if self._gripper_abort.is_set():
                raise GripperAborted()
            remaining = end - time.time()
            if remaining <= 0:
                return
            time.sleep(min(GRIPPER_ABORT_POLL_SEC, remaining))

    def _enter_gripper(self):
        self._abort_running_gripper_thread()
        self._set_state(MissionState.GRIPPER)
        self.gripper_phase = GripperPhase.ENABLE_PID
        self.gripper_angle = 0
        self.gripper_last_enc = None
        self.gripper_stable_count = 0
        self.tcp.send("STOP")
        self.tcp.reset_encoders()
        self._gripper_thread = threading.Thread(
            target=self._gripper_sequence, daemon=True, name="GripperSequence")
        self._gripper_thread.start()

    def _gripper_sequence(self):
        try:
            self.gripper_phase = GripperPhase.ENABLE_PID
            self.tcp.enable_gripper_pid()
            self._abortable_sleep(0.1)

            self.gripper_phase = GripperPhase.HOMING
            self._home_gripper()

            self.gripper_phase = GripperPhase.RESET
            self.tcp.send("RESET")
            self.log("[MISSION] Gripper mechanically reset - zeroed")
            self.gripper_angle = 0
            self._abortable_sleep(0.2)

            self.gripper_phase = GripperPhase.RAMP_UP
            self._ramp_gripper(0, GRIPPER_RAMP_MAX_ANGLE)

            self.gripper_phase = GripperPhase.HOLD_MAX
            self.log(f"[MISSION] Holding at {GRIPPER_RAMP_MAX_ANGLE} deg for "
                     f"{self.gripper_hold_sec}s")
            self._abortable_sleep(self.gripper_hold_sec)

            self.gripper_phase = GripperPhase.RAMP_DOWN
            self._ramp_gripper(GRIPPER_RAMP_MAX_ANGLE, 0)

            self.gripper_phase = GripperPhase.DONE
            self.log("[MISSION] Gripper sequence complete -> awaiting user "
                     "confirmation to RESET and repeat")
            self._set_state(MissionState.AWAITING_RESET)
        except GripperAborted:
            self.log("[MISSION] Gripper sequence stopped (E-STOP or overridden)")
            self.gripper_phase = None
        except Exception as e:
            self.log(f"[MISSION] Gripper sequence error: {e}")
            self._set_state(MissionState.ERROR)

    def _home_gripper(self):
        angle = 0
        last_reading = None
        stable_count = 0
        last_step_time = 0.0
        last_poll_time = 0.0
        start_time = time.time()
        while True:
            if self._gripper_abort.is_set():
                raise GripperAborted()
            now = time.time()
            if now - start_time > GRIPPER_HOMING_TIMEOUT_SEC:
                self.log("[MISSION] Homing timeout - proceeding to RESET anyway "
                         "(check the gripper hardware!)")
                return
            if abs(angle) > GRIPPER_HOMING_MAX_ANGLE_MAG:
                self.log("[MISSION] Homing exceeded safety angle - aborting "
                         "homing, proceeding to RESET (check hardware!)")
                return
            if now - last_step_time >= GRIPPER_HOMING_STEP_INTERVAL:
                angle -= GRIPPER_HOMING_STEP_DEG
                self.gripper_angle = angle
                self.tcp.set_gripper_angle(angle)
                last_step_time = now
            if now - last_poll_time >= GRIPPER_HOMING_POLL_INTERVAL:
                reply = self.tcp.query_gripper_encoder()
                last_poll_time = now
                reading = _parse_int(reply)
                if reading is not None:
                    self.gripper_last_enc = reading
                    if (last_reading is not None and
                            abs(reading - last_reading) <= GRIPPER_HOMING_STABLE_TOL):
                        stable_count += 1
                    else:
                        stable_count = 0
                    last_reading = reading
                    self.gripper_stable_count = stable_count
                    if stable_count >= GRIPPER_HOMING_STABLE_COUNT:
                        self.log(f"[MISSION] Encoder stable at {reading} - "
                                 "mechanical limit reached")
                        return
            time.sleep(0.02)

    def _ramp_gripper(self, start, end):
        direction = 1 if end > start else -1
        angle = start
        step_mag = max(1, self.gripper_ramp_step)
        while (direction > 0 and angle < end) or (direction < 0 and angle > end):
            if self._gripper_abort.is_set():
                raise GripperAborted()
            angle += direction * step_mag
            if (direction > 0 and angle > end) or (direction < 0 and angle < end):
                angle = end
            self.gripper_angle = angle
            self.tcp.set_gripper_angle(angle)
            self._abortable_sleep(self.gripper_ramp_interval)
        if self.gripper_angle != end:
            self.gripper_angle = end
            self.tcp.set_gripper_angle(end)

    # ------------------------------------------------------------------ #
    def _run(self):
        while not self._stop:
            frame = self.reader.read_frame()
            if frame is None:
                continue
            obs = None
            try:
                fh, fw = frame.shape[:2]
                self._ensure_intrinsics_for(fw, fh)
                if UNDISTORT_TAG_FRAME and self._map1 is not None:
                    frame = cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                obs = self._best_observation(gray)
            except Exception:
                self.log("[MISSION] Frame processing raised an exception; skipping frame")

            if not self._estop.is_set():
                handler = self._dispatch.get(self.state)
                if handler is not None:
                    handler(frame, obs)

            self._draw_overlay(frame, obs)
            self._draw_hud(frame)

            if self.frame_q.full():
                try:
                    self.frame_q.get_nowait()
                except queue.Empty:
                    pass
            self.frame_q.put(frame)

        self.log("[MISSION] Runner exited.")


