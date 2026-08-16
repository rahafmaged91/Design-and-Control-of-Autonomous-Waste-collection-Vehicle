"""
control/lane_keep_controller.py
=================================
The autonomous lane-keeping loop: grabs frames from the local camera,
runs them through LaneDetector + LaneTopologyModel, turns the result
into a PID-controlled steering command (with slew-rate limiting and
first-image geometry correction), and sends it to the vehicle over
TCP. This is "STATE 1" (LANE_FOLLOWING) of the mission state machine.
"""

import math
import threading
import time

import cv2

from config import (
    clamp, STEER_CENTER, STEER_MAX_RIGHT, STEER_MAX_LEFT,
    STEER_RATE_LIMIT, STEER_RATE_WINDOW_S, SPEED_MAX_VALUE,
)
from camera_utils import open_camera, CV2_AVAILABLE
from control.pid import PIDController
from vision.lane_detector import LaneDetector
from vision.lane_topology import LaneTopologyModel

# ─── Lane keeping controller (camera loop + PID + servo slew limit) ──────────

class LaneKeepController:
    """
    Owns the camera + detection loop and drives a VehicleController
    directly (the manual key loop stands down while active).

    Two ROAD TYPES (color palettes) are supported, selected by the user at
    startup via `road_type`:

      * "2_black_border" - the original pipeline: black border detection
        (+ optional yellow divider) -> LaneTopologyModel -> multi-lane
        world model -> PID to the lane target (see module docstring).

      * "yellow_follower" - a simpler, dedicated pipeline for a road whose
        only marking is a single painted YELLOW center/divider line (no
        black borders relied upon). The vehicle keeps that yellow line
        inside an adjustable REGION (width + horizontal position, both
        live GUI sliders) instead of driving to a single target point.

    Priority-aware degradation (2_black_border):
      TRACKING    both required boundaries real & seen -> normal control.
      PREDICTING  (black_yellow lane only) yellow seen, black edge not ->
                  target already comes from the yellow line + the learned
                  offset (topology model); mild speed trim only.
      RECOVERING  the CRITICAL boundary (yellow in a black_yellow lane,
                  either edge in a 2_blacks lane) is gone -> steer toward
                  its last remembered side, slow down more, until it's
                  visible again. All boundaries gone -> steer toward the
                  remembered HOME LANE anchor (the specific lane the
                  vehicle started in - see LaneTopologyModel.home_lane_width).
      LANE_LOST   no memory left / recovery timed out -> "STOP".

    Degradation (yellow_follower):
      TRACKING    the yellow line is seen this frame -> region deadband
                  control.
      RECOVERING  the yellow line is not seen this frame, but was seen
                  recently (short-term memory of its last x) -> steer
                  back toward its last remembered side, slow down.
      LANE_LOST   no memory left / recovery timed out -> "STOP".
    """

    ROAD_TYPE_2_BLACK   = "2_black_border"
    ROAD_TYPE_YELLOW    = "yellow_follower"

    # how strongly the first-image geometry violation pulls the error,
    # and the extra steering bias applied toward a faded line side
    GEO_GAIN            = 1.5
    GEO_ERR_CLAMP       = 0.45
    RECOVERY_BIAS       = 0.20
    PREDICT_BIAS        = 0.0     # target already yellow-anchored; no extra bias
    RECOVERY_SPEED_FACT = 0.45    # of cruise speed while recovering (critical)
    PREDICT_SPEED_FACT  = 0.85    # of cruise speed while predicting (mild)

    def __init__(self, vehicle_ctrl, camera_index=0, log_cb=print,
                 frame_cb=None, confirm_cb=None, params_cb=None,
                 loop_delay=0.03, capture_width=400, capture_height=300,
                 road_type=None):
        self.vc = vehicle_ctrl
        self.camera_index = camera_index
        self.log_cb = log_cb
        self.frame_cb = frame_cb        # (debug_frame_bgr, info_dict)
        self.confirm_cb = confirm_cb    # (topology_info) -> ask user once
        self.params_cb = params_cb      # () -> dict of live GUI parameters
        self.loop_delay = loop_delay
        self.capture_width = capture_width
        self.capture_height = capture_height

        # ROAD TYPE (color palette) selected by the user at startup.
        self.road_type = road_type or self.ROAD_TYPE_2_BLACK

        self.detector = LaneDetector()
        self.model = LaneTopologyModel()
        self.pid = PIDController()

        self.cap = None
        self.running = False
        self.thread = None

        self.lost_frame_limit = 10
        self.recovery_frame_limit = 60    # ~2 s of steer-back attempts
        self._lost_count = 0
        self._recover_count = 0           # blackout (all-lanes-lost) budget ONLY
        self._recover_logged = False      # blackout recovery log latch
        self._deg_state_logged = None     # last logged in-lane degraded state
        self._stable_count = 0
        self._stop_sent = False

        # first-image calibration: system type (2_blacks / black_yellow)
        # plus lane width + external spaces geometry. (2_black_border only)
        self.system_type = None           # "2_blacks" | "black_yellow"
        self.calib = None                 # {"lane_w","ext_left","ext_right","frame_w"}
        self._calib_pending = False

        # user must confirm the detected lane before the vehicle moves
        # (2_black_border only - the yellow_follower road type has no
        # multi-lane topology to confirm, so it starts driving right away)
        self.awaiting_confirmation = (self.road_type == self.ROAD_TYPE_2_BLACK)
        self._confirm_requested = False

        # slew-rate limiter memory: starts at the CENTER value per spec
        self._steer_prev = STEER_CENTER
        self._last_sent_steer = None

        # -- yellow_follower state (short-term memory of the line's x,
        #    independent of the LaneTopologyModel used by 2_black_border) --
        self._yf_last_x = None
        self._yf_lost_count = 0
        self._yf_recover_count = 0
        self._yf_state = LaneTopologyModel.STATE_INIT

    # -- lifecycle ---------------------------------------------------------------

    def start(self):
        if not CV2_AVAILABLE:
            self.log_cb("Lane tracking requires opencv-python and numpy "
                        "(pip install opencv-python numpy).")
            return False
        self.cap = open_camera(self.camera_index, self.log_cb)
        if self.cap is None:
            self.log_cb(f"Could not open camera index {self.camera_index}.")
            return False

        self.running = True
        self._lost_count = 0
        self._recover_count = 0
        self._recover_logged = False
        self._deg_state_logged = None
        self._stable_count = 0
        self._stop_sent = False
        self.system_type = None
        self.calib = None
        self._calib_pending = False
        self.awaiting_confirmation = (self.road_type == self.ROAD_TYPE_2_BLACK)
        self._confirm_requested = False
        self._steer_prev = STEER_CENTER
        self._last_sent_steer = None
        self._yf_last_x = None
        self._yf_lost_count = 0
        self._yf_recover_count = 0
        self._yf_state = LaneTopologyModel.STATE_INIT
        self.pid.reset()
        self.model.reset()
        self.detector.reset_temporal_filter()

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        if self.road_type == self.ROAD_TYPE_YELLOW:
            self.log_cb(f"Lane tracking started (camera {self.camera_index}, "
                        "road type: YELLOW FOLLOWER). Driving to keep the "
                        "yellow line inside the selected region...")
        else:
            self.log_cb(f"Lane tracking started (camera {self.camera_index}, "
                        "road type: 2 BLACK BORDERS). Waiting for a stable "
                        "lane detection to confirm...")
        return True

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
            self.thread = None
        if self.cap:
            self.cap.release()
            self.cap = None
        self._send_stop()
        self.vc.speed = 0
        self.vc.steer = self.vc.steer_center
        self.vc._send_commands()
        self.log_cb("Lane tracking stopped.")

    # -- user confirmation ---------------------------------------------------------

    def confirm_lane(self, shift=0):
        """Called from the GUI: 0 = detected lane is correct,
        -1/+1 = the vehicle should actually track the neighbouring lane.
        The next stable frame after confirmation is used as the FIRST IMAGE
        for calibration: the lane's SYSTEM TYPE (2_blacks / black_yellow)
        plus the lane width + external spaces geometry, and seeds the
        dedicated HOME LANE width memory.
        (Only used by the 2_black_border road type.)"""
        if shift != 0:
            ok = self.model.shift_desired_lane(shift, self.capture_width)
            if not ok:
                self.log_cb("Cannot shift: lane width not learned yet.")
        self.awaiting_confirmation = False
        self._calib_pending = True        # calibrate on the first solid frame
        self.pid.reset()
        self.log_cb("Lane confirmed - driving." if shift == 0 else
                    f"Lane shifted ({'left' if shift < 0 else 'right'}) - driving.")

    # -- helpers ---------------------------------------------------------------------

    def _send_stop(self):
        if self.vc.tcp.connected:
            self.vc.tcp.send("STOP")

    def _apply_params(self):
        if not self.params_cb:
            return {}
        p = self.params_cb()
        d = self.detector
        d.black_thresh  = p.get("black_thresh", d.black_thresh)
        d.roi_top_ratio = p.get("roi_top_ratio", d.roi_top_ratio)
        d.min_line_w    = p.get("min_line_w", d.min_line_w)
        d.max_line_w    = p.get("max_line_w", d.max_line_w)
        d.num_rows      = int(clamp(p.get("num_rows", d.num_rows), 4, 48))
        d.lpf_alpha     = clamp(p.get("lpf_alpha", d.lpf_alpha), 0.05, 1.0)
        d.yellow_sat_min = int(clamp(p.get("yellow_sat_min", d.yellow_sat_min),
                                     0, 255))
        d.yellow_val_min = int(clamp(p.get("yellow_val_min", d.yellow_val_min),
                                     0, 255))
        self.model.target_offset = p.get("target_offset", 0.0)
        self.model.memory_frames = int(clamp(p.get("memory_frames",
                                             self.model.memory_frames), 5, 120))
        self.pid.kp = p.get("kp", self.pid.kp)
        self.pid.ki = p.get("ki", self.pid.ki)
        self.pid.kd = p.get("kd", self.pid.kd)
        self.pid.deadband = p.get("deadband", self.pid.deadband)
        return p

    def _slew_limited(self, steer_target, dt):
        """Enforce the mechanical limit: at most 160 servo units / 50 ms.
        Remembers the previously commanded value (starting at center)."""
        max_step = STEER_RATE_LIMIT * (dt / STEER_RATE_WINDOW_S)
        delta = steer_target - self._steer_prev
        if abs(delta) > max_step:
            steer_target = self._steer_prev + math.copysign(max_step, delta)
        self._steer_prev = steer_target
        return int(round(steer_target))

    def _steer_from_u(self, u, dt):
        """Map PID output u [-1..1] onto the asymmetric servo range around
        center and slew-limit it. +u (target to the RIGHT of view center)
        -> steer right -> LOWER servo value."""
        if u >= 0:
            steer_target = STEER_CENTER - u * (STEER_CENTER - STEER_MAX_RIGHT)
        else:
            steer_target = STEER_CENTER - u * (STEER_MAX_LEFT - STEER_CENTER)
        steer_target = clamp(steer_target, STEER_MAX_RIGHT, STEER_MAX_LEFT)
        return self._slew_limited(steer_target, dt)

    # -- first-image calibration (system type + geometry) -----------------------------

    def _maybe_calibrate(self, topo, frame_w):
        """Capture the FIRST-IMAGE calibration on a frame where the chosen
        lane's two boundaries are both real and currently seen:
          - SYSTEM TYPE: "2_blacks" or "black_yellow" (from topo['lane_type'])
          - LANE WIDTH + EXTERNAL SPACES geometry (anti lane-fading)
          - HOME LANE WIDTH memory (seeds LaneTopologyModel.home_lane_width
            so re-synthesis/recovery has a trustworthy value from frame 1,
            anchored to the specific lane the vehicle is starting in)."""
        if not self._calib_pending:
            return
        if topo["missing_side"] != 0 or topo["desired_lane"] is None:
            return
        if topo["lane_type"] not in ("black_yellow", "2_blacks"):
            return
        l, r = topo["lanes"][topo["desired_lane"]]
        self.system_type = topo["lane_type"]
        self.model.lock_system_type(self.system_type)
        self.calib = {"lane_w": float(r - l),
                      "ext_left": float(l),
                      "ext_right": float(frame_w - r),
                      "frame_w": float(frame_w)}
        self.model.seed_home_lane(self.calib["lane_w"], (l + r) / 2.0)
        self._calib_pending = False
        if self.system_type == "black_yellow":
            side_txt = ("yellow on the LEFT, black on the RIGHT"
                       if topo["yellow_x"] < topo["black_x"]
                       else "yellow on the RIGHT, black on the LEFT")
            self.log_cb(f"Calibrated system type: BLACK_YELLOW ({side_txt}). "
                        f"Lane width {self.calib['lane_w']:.0f}px (home-lane "
                        "memory seeded). The vehicle will now drive by "
                        "FOLLOWING THE YELLOW LINE directly (target = "
                        "yellow line position); losing the yellow line is "
                        "critical and triggers recovery steering back "
                        "toward it, anchored to this lane's own width even "
                        "if both borders leave view.")
        else:
            self.log_cb(f"Calibrated system type: 2_BLACKS. Lane width "
                        f"{self.calib['lane_w']:.0f}px (home-lane memory "
                        f"seeded), external space L {self.calib['ext_left']:.0f}px "
                        f"/ R {self.calib['ext_right']:.0f}px.")

    def _geometry_correction(self, topo, frame_w, params):
        """If the visible lane width has SHRUNK while an external space has
        GROWN beyond the fair error margin (both relative to the calibrated
        lane width), return an error term that steers the vehicle back until
        the first-image distances are restored. 0.0 while within margin."""
        if self.system_type == "black_yellow":
            return 0.0
        if self.calib is None or topo["desired_lane"] is None:
            return 0.0
        l, r = topo["lanes"][topo["desired_lane"]]
        lane_w = r - l
        ext_l = l
        ext_r = frame_w - r
        margin = clamp(params.get("calib_margin", 0.12), 0.02, 0.6)
        tol = margin * self.calib["lane_w"]

        lane_shrunk = (self.calib["lane_w"] - lane_w) > tol
        ext_grew = ((ext_l - self.calib["ext_left"]) > tol or
                    (ext_r - self.calib["ext_right"]) > tol)
        if not (lane_shrunk and ext_grew):
            return 0.0

        geo = ((ext_l - self.calib["ext_left"]) -
               (ext_r - self.calib["ext_right"])) / frame_w
        return clamp(self.GEO_GAIN * geo, -self.GEO_ERR_CLAMP,
                     self.GEO_ERR_CLAMP)

    # -- yellow_follower region helper --------------------------------------------------

    @staticmethod
    def region_bounds(frame_w, width_pct, offset_pct):
        """Turn the two GUI sliders into a pixel [lo, hi] band, clamped so
        it always stays fully inside the camera frame (horizontally)."""
        width_pct = clamp(width_pct, 5.0, 100.0)
        offset_pct = clamp(offset_pct, -100.0, 100.0)
        width_px = (width_pct / 100.0) * frame_w
        width_px = clamp(width_px, 4.0, float(frame_w))
        center = frame_w / 2.0 + (offset_pct / 100.0) * (frame_w / 2.0)
        lo = center - width_px / 2.0
        hi = center + width_px / 2.0
        if lo < 0:
            hi -= lo
            lo = 0.0
        if hi > frame_w:
            lo -= (hi - frame_w)
            hi = float(frame_w)
        lo = clamp(lo, 0.0, frame_w)
        hi = clamp(hi, 0.0, frame_w)
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi

    # -- main loop ------------------------------------------------------------------

    def _loop(self):
        prev_t = time.time()
        while self.running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            params = self._apply_params()
            if params.get("flip_horizontal"):
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            if (w, h) != (self.capture_width, self.capture_height):
                frame = cv2.resize(frame,
                                   (self.capture_width, self.capture_height))

            if frame.std() < 3:
                time.sleep(0.05)
                continue

            now = time.time()
            dt = clamp(now - prev_t, 1e-3, 0.2)
            prev_t = now

            det = self.detector.process(frame)

            if self.road_type == self.ROAD_TYPE_YELLOW:
                self._loop_yellow_follower(det, dt, params)
            else:
                self._loop_2_black_border(det, dt, params)

            time.sleep(self.loop_delay)

    # -- 2_black_border branch ----------------------------------------------------

    def _loop_2_black_border(self, det, dt, params):
        topo = self.model.update(det["boundary_xs"], det["frame_w"])

        info = {"det": det, "topo": topo, "steer": self._steer_prev,
                "error": 0.0, "pid_out": 0.0, "geo_err": 0.0,
                "awaiting": self.awaiting_confirmation}

        if topo["target_x"] is None:
            if (not self.awaiting_confirmation
                    and topo.get("recovery_x") is not None
                    and self._recover_count < self.recovery_frame_limit):
                self._lost_count = 0
                self._recover(topo, det, dt, params, info)
            else:
                self._handle_lost()
        else:
            self._lost_count = 0
            self._stop_sent = False
            if topo["missing_side"] == 0:
                self._recover_count = 0
                self._recover_logged = False
                self._deg_state_logged = None
            if self.awaiting_confirmation:
                self._handle_confirmation_phase(topo)
            else:
                self._maybe_calibrate(topo, det["frame_w"])
                self._drive(topo, det, dt, params, info)

        if self.frame_cb:
            dbg = self._draw_topology(det["debug_frame"], det, topo, info)
            self.frame_cb(dbg, info)

    # -- yellow_follower branch ------------------------------------------------

    def _loop_yellow_follower(self, det, dt, params):
        w = det["frame_w"]
        cx = w / 2.0

        yellow_tracks = [t for t in det["boundaries"] if t["color"] == "yellow"]
        chosen = None
        if yellow_tracks:
            if self._yf_last_x is not None:
                chosen = min(yellow_tracks,
                            key=lambda t: abs(t["x_ctrl"] - self._yf_last_x))
            else:
                chosen = max(yellow_tracks, key=lambda t: len(t["xs"]))

        lo, hi = self.region_bounds(w, params.get("yf_region_width_pct", 40.0),
                                    params.get("yf_region_offset_pct", 0.0))

        info = {"det": det, "topo": None, "steer": self._steer_prev,
                "error": 0.0, "pid_out": 0.0, "geo_err": 0.0,
                "awaiting": False,
                "yf": {"region": (lo, hi), "x": None, "state": self._yf_state}}

        if chosen is not None:
            x = chosen["x_ctrl"]
            self._yf_last_x = x
            self._yf_lost_count = 0
            self._yf_recover_count = 0
            self._yf_state = LaneTopologyModel.STATE_TRACKING
            self._stop_sent = False

            if x < lo:
                err_px = x - lo
            elif x > hi:
                err_px = x - hi
            else:
                err_px = 0.0
            error_norm = clamp(err_px / (w / 2.0), -1.0, 1.0)

            u = self.pid.update(error_norm, dt)
            if params.get("invert_steering"):
                u = -u
            steer = self._steer_from_u(u, dt)

            cruise = int(params.get("cruise_speed", 350))
            slowdown = clamp(params.get("max_turn_slowdown", 0.5), 0.0, 0.9)
            speed = int(clamp(cruise * (1 - slowdown * abs(u)), 0, SPEED_MAX_VALUE))

            self.vc.speed = speed
            self.vc.steer = steer
            self.vc._send_commands()

            info["error"] = error_norm
            info["pid_out"] = u
            info["steer"] = steer
            info["yf"]["x"] = x
            info["yf"]["state"] = self._yf_state
        else:
            self._yf_lost_count += 1
            if (self._yf_last_x is not None
                    and self._yf_recover_count < self.recovery_frame_limit):
                self._yf_recover_count += 1
                self._yf_state = LaneTopologyModel.STATE_RECOVERING
                if self._deg_state_logged != "YF_RECOVERING":
                    side = "left" if self._yf_last_x < cx else "right"
                    self.log_cb(f"Yellow line lost - recovering: steering "
                                f"{side} toward its last remembered "
                                "position...")
                    self._deg_state_logged = "YF_RECOVERING"

                if self._yf_last_x < lo:
                    err_px = self._yf_last_x - lo
                elif self._yf_last_x > hi:
                    err_px = self._yf_last_x - hi
                else:
                    err_px = self._yf_last_x - cx
                error_norm = clamp(err_px / (w / 2.0), -1.0, 1.0)
                u = self.pid.update(error_norm, dt)
                if params.get("invert_steering"):
                    u = -u
                steer = self._steer_from_u(u, dt)

                cruise = int(params.get("cruise_speed", 350))
                speed = int(clamp(cruise * self.RECOVERY_SPEED_FACT,
                                  0, SPEED_MAX_VALUE))
                self.vc.speed = speed
                self.vc.steer = steer
                self.vc._send_commands()

                info["error"] = error_norm
                info["pid_out"] = u
                info["steer"] = steer
                info["yf"]["state"] = self._yf_state
            else:
                self._yf_state = LaneTopologyModel.STATE_LOST
                self._handle_lost()
                info["yf"]["state"] = self._yf_state
                self._deg_state_logged = None

        if self.frame_cb:
            dbg = self._draw_yellow_follower(det["debug_frame"], det, info)
            self.frame_cb(dbg, info)

    def _handle_lost(self):
        self._lost_count += 1
        self._stable_count = 0
        self.pid.reset()
        if self._lost_count >= self.lost_frame_limit:
            if not self._stop_sent:
                self.log_cb("No lanes detected and no usable memory - "
                            "sending STOP.")
                self._stop_sent = True
            self._send_stop()
            self.vc.speed = 0
            self.vc.steer = self.vc.steer_center
            self._steer_prev = self.vc.steer_center

    def _recover(self, topo, det, dt, params, info):
        self._recover_count += 1
        self._stable_count = 0
        if not self._recover_logged:
            side = "left" if topo["missing_side"] < 0 else "right"
            self.log_cb(f"Lane view lost - recovering: steering {side} "
                        "toward the remembered home lane...")
            self._recover_logged = True

        w = det["frame_w"]
        cx = w / 2.0
        error_norm = clamp((topo["recovery_x"] - cx) / (w / 2.0), -1.0, 1.0)
        u = self.pid.update(error_norm, dt)
        if params.get("invert_steering"):
            u = -u
        steer = self._steer_from_u(u, dt)

        cruise = int(params.get("cruise_speed", 350))
        speed = int(clamp(cruise * self.RECOVERY_SPEED_FACT,
                          0, SPEED_MAX_VALUE))

        self.vc.speed = speed
        self.vc.steer = steer
        self.vc._send_commands()

        info["error"] = error_norm
        info["pid_out"] = u
        info["steer"] = steer

    def _handle_confirmation_phase(self, topo):
        self._send_stop()
        self.vc.speed = 0
        if (topo["missing_side"] == 0
                and topo["lane_type"] in ("black_yellow", "2_blacks")):
            self._stable_count += 1
        else:
            self._stable_count = max(0, self._stable_count - 1)
        if self._stable_count >= 8 and not self._confirm_requested:
            self._confirm_requested = True
            if self.confirm_cb:
                self.confirm_cb(dict(topo))
            else:
                self.confirm_lane(0)

    def _drive(self, topo, det, dt, params, info):
        w = det["frame_w"]
        cx = w / 2.0
        error_norm = (topo["target_x"] - cx) / (w / 2.0)

        geo_err = self._geometry_correction(topo, w, params)
        error_norm += geo_err

        state = topo["state"]
        recovering = state == LaneTopologyModel.STATE_RECOVERING
        predicting = state == LaneTopologyModel.STATE_PREDICTING

        if recovering and topo["missing_side"] != 0:
            error_norm += self.RECOVERY_BIAS * topo["missing_side"]
            if self._deg_state_logged != "RECOVERING":
                side = "left" if topo["missing_side"] < 0 else "right"
                what = ("yellow line" if self.system_type == "black_yellow"
                        else "border line")
                self.log_cb(f"Critical: {what} fading on the {side} - "
                            f"steering {side} to bring it back into view...")
                self._deg_state_logged = "RECOVERING"
        elif predicting:
            if self._deg_state_logged != "PREDICTING":
                self.log_cb("Black edge faded, yellow line still visible - "
                            "predicting target from the yellow line...")
                self._deg_state_logged = "PREDICTING"

        error_norm = clamp(error_norm, -1.0, 1.0)
        u = self.pid.update(error_norm, dt)
        if params.get("invert_steering"):
            u = -u

        steer = self._steer_from_u(u, dt)

        cruise = int(params.get("cruise_speed", 350))
        slowdown = clamp(params.get("max_turn_slowdown", 0.5), 0.0, 0.9)
        speed = cruise * (1 - slowdown * abs(u))
        if recovering:
            speed *= 0.6
        if predicting:
            speed *= self.PREDICT_SPEED_FACT
        speed = int(clamp(speed, 0, SPEED_MAX_VALUE))

        self.vc.speed = speed
        self.vc.steer = steer
        if self._last_sent_steer is None or steer != self._last_sent_steer:
            self._last_sent_steer = steer
        self.vc._send_commands()

        info["error"] = error_norm
        info["pid_out"] = u
        info["geo_err"] = geo_err
        info["steer"] = steer

    # -- debug drawing -----------------------------------------------------------------

    def _draw_topology(self, debug, det, topo, info):
        h, w = debug.shape[:2]
        cy = det["control_y"]
        for (l, r), name in zip(topo["lanes"], topo["lane_names"]):
            mid = int((l + r) / 2)
            cv2.line(debug, (int(l), cy - 14), (int(r), cy - 14),
                     (0, 255, 120), 2)
            cv2.putText(debug, name, (max(2, mid - 40), cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 120), 1)
        for x, conf, virtual, color in topo["boundaries"]:
            base = (0, 255, 255) if color == "yellow" else (255, 0, 0)
            cv2.circle(debug, (int(x), cy), 5, base, 2 if virtual else -1)
        if topo["target_x"] is not None:
            tx = int(topo["target_x"])
            cv2.line(debug, (tx, det["roi_top"]), (tx, cy), (0, 255, 0), 2)
        elif topo.get("recovery_x") is not None:
            rx = int(clamp(topo["recovery_x"], 0, w - 1))
            cv2.line(debug, (rx, det["roi_top"]), (rx, cy), (255, 0, 255), 2)
            cv2.putText(debug, "recover", (max(2, rx - 30), det["roi_top"] + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 255), 1)

        if self.calib is not None:
            cl = int(self.calib["ext_left"])
            cr = int(self.calib["frame_w"] - self.calib["ext_right"])
            for x in (cl, cr):
                cv2.line(debug, (x, cy + 4), (x, cy + 12), (0, 255, 255), 2)

        sys_txt = f" | {self.system_type}" if self.system_type else ""
        status = topo["state"] + sys_txt + (" | CONFIRM LANE?" if info["awaiting"] else "")
        cv2.putText(debug, status, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1)
        home_w = topo.get("home_lane_width")
        home_txt = f"{home_w:.0f}px" if home_w else "-"
        cv2.putText(debug,
                    f"pos:{topo['position']}  type:{topo['lane_type']}  "
                    f"home_w:{home_txt}  err:{info['error']:+.2f}  "
                    f"geo:{info.get('geo_err', 0):+.2f}  steer:{info['steer']}",
                    (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (200, 200, 200), 1)
        return debug

    def _draw_yellow_follower(self, debug, det, info):
        h, w = debug.shape[:2]
        cy = det["control_y"]
        top = det["roi_top"]
        lo, hi = info["yf"]["region"]

        REGION_COLOR = (0, 255, 0)
        cv2.line(debug, (int(lo), top), (int(lo), cy), REGION_COLOR, 2)
        cv2.line(debug, (int(hi), top), (int(hi), cy), REGION_COLOR, 2)
        cv2.line(debug, (int(lo), cy - 6), (int(hi), cy - 6), REGION_COLOR, 1)
        cv2.putText(debug, "region", (max(2, int(lo)), top + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, REGION_COLOR, 1)

        x = info["yf"]["x"]
        if x is not None:
            cv2.circle(debug, (int(x), cy), 6, (0, 255, 255), -1)
            cv2.line(debug, (int(x), top), (int(x), cy), (0, 255, 255), 2)
        elif self._yf_last_x is not None:
            lx = int(clamp(self._yf_last_x, 0, w - 1))
            cv2.line(debug, (lx, top), (lx, cy), (255, 0, 255), 2)
            cv2.putText(debug, "last seen", (max(2, lx - 30), top + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        cv2.line(debug, (w // 2, top), (w // 2, cy), (180, 180, 180), 1)

        status = f"YELLOW_FOLLOWER | {info['yf']['state']}"
        cv2.putText(debug, status, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1)
        cv2.putText(debug,
                    f"region:[{lo:.0f},{hi:.0f}]  x:{('-' if x is None else f'{x:.0f}')}  "
                    f"err:{info['error']:+.2f}  steer:{info['steer']}",
                    (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (200, 200, 200), 1)
        return debug

