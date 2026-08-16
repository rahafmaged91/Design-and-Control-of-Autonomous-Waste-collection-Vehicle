"""
config.py
=========
Central place for every tunable constant used across the project:
steering/servo limits, the ESP32-CAM stream, AprilTag calibration,
gripper timing, and the shared clamp() helper.

Splitting these out means every other module can do
`from config import STEER_CENTER, clamp, ...` instead of the constants
being scattered through one giant file.
"""

import math

# ─── Steering / protocol constants ──────────────────────────────────────────

STEER_MAX_RIGHT   = 1200      # servo value, hard right
STEER_MAX_LEFT    = 4000      # servo value, hard left
STEER_CENTER      = 2400      # servo value, straight
SPEED_MAX_VALUE   = 1023
TCP_DEFAULT_PORT  = 4545

# Mechanical limit: the steering value must not change faster than
# 160 units per 50 ms.
STEER_RATE_LIMIT     = 160
STEER_RATE_WINDOW_S  = 0.05


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


# ─── STATE 2 / STATE 3: AprilTag centering + gripper (port 8080 stream) ─────
#
# STATE 1  LANE_FOLLOWING   : the existing multi-lane tracker (above) drives
#                              the vehicle while, in parallel, this app
#                              watches the ESP32-CAM MJPEG stream (TCP/HTTP
#                              port 8080) for an AprilTag, and the PC camera
#                              feed for a stop sign (stop-sign CV is not yet
#                              implemented - see detect_stop_sign() below).
# STATE 2a CENTERING         : (AprilTag stop) PID-centers the vehicle on the
#                              tag using the 8080 stream.
# STATE 2b STOPSIGN_HALT     : (stop-sign stop) a fixed momentary halt, then
#                              lane following resumes automatically.
# STATE 3  GRIPPER            : (AprilTag branch only) grabs the bin.
# AWAITING_RESET              : after the gripper sequence finishes, the
#                              system waits for the user to confirm before
#                              sending "RESET" (port 4545) and repeating.

# --- ESP32-CAM MJPEG stream (always live once the app starts) --------------
ESP32_CAM_IP      = "192.168.137.123"
ESP32_CAM_PORT    = 8080
ESP32_STREAM_PATH = "/video"

RESOLUTION_UDP_PORT = 8888
FRAMESIZE_PRESETS = {
    "QQVGA 160x120":  1,
    "QCIF 176x144":   2,
    "HQVGA 240x176":  3,
    "QVGA 320x240":   4,
    "CIF 400x296":    5,
    "HVGA 480x320":   6,
    "VGA 640x480":    7,
    "SVGA 800x600":   8,
    "XGA 1024x768":   9,
    "UXGA 1600x1200": 10,
}

# --- STATE 1 (searching) confirmation -----------------------------------
TAG_CONFIRM_FRAMES = 2     # consecutive hit frames required to confirm the tag

# --- AprilTag camera calibration (at CALIB_WIDTH x CALIB_HEIGHT), scaled ---
TAG_CALIB_WIDTH   = 320
TAG_CALIB_HEIGHT  = 240
TAG_SIZE_METERS   = 0.054
TAG_FOCAL_LENGTH_X = 110.59
TAG_FOCAL_LENGTH_Y = 113.71
TAG_CENTER_X       = 160.07
TAG_CENTER_Y       = 123.49

UNDISTORT_TAG_FRAME = True
TAG_DETECTION_SCALE = 1.0
MJPEG_CHUNK_SIZE     = 65536
MJPEG_QUEUE_SIZE     = 2
TAG_MIN_MARGIN       = 15

TAG_CLAHE_CLIP_LIMIT = 3.0
TAG_CLAHE_GRID_SIZE  = (8, 8)
TAG_DENOISE_D            = 5
TAG_DENOISE_SIGMA_COLOR  = 50
TAG_DENOISE_SIGMA_SPACE  = 50

# --- STATE 2a - PID centering on the AprilTag -------------------------------
TAG_TRANSITION_DELAY_SEC = 5.0    # fixed pause between tag-confirm and centering

WHEEL_DIAMETER_MM      = 61.0     # vehicle wheel diameter -> mm to degrees
WHEEL_CIRCUMFERENCE_MM = math.pi * WHEEL_DIAMETER_MM

# Startup defaults for the centering PID - the ONLY supported way to change
# these at runtime is the "AprilTag PID" panel on the new GUI tab, via
# MissionRunner.update_centering_params().
TAG_PID_KP = 0.40
TAG_PID_KI = 0.03
TAG_PID_KD = 0.20

TAG_ERROR_MARGIN_MM = 8.0         # deadband: offsets smaller than this -> 0
TAG_PID_OUT_MIN_MM  = -150.0
TAG_PID_OUT_MAX_MM  = 150.0
TAG_MAX_CMD_DEGREES = 360         # hard safety clamp on any single SET command
TAG_MIN_CMD_DELTA_DEG = 1         # only resend SET if it changed by >= this
TAG_CENTER_CMD_HZ    = 10.0
TAG_CENTER_CMD_INTERVAL = 1.0 / TAG_CENTER_CMD_HZ

TAG_LOST_HOLD_TIMEOUT = 1.5       # seconds of no tag before holding position
TAG_CENTER_LOCK_FRAMES = 15       # consecutive in-margin frames -> "locked"
TAG_CENTER_LOCK_MISS_PENALTY = 3  # single noisy out-of-margin frame only
                                   # costs this many "lock" frames instead of
                                   # wiping the whole streak, so one bad
                                   # detection doesn't force 15 more in a row

PID_ENABLE_VALUE = 10             # "PID1 10"/"PID2 10" enables drive PID mode

# --- STOPSIGN branch (state 2b) - STILL TO BE IMPLEMENTED ------------------
STOPSIGN_HALT_SEC = 5.0           # momentary stop duration once implemented

# --- STATE 3 - Gripper (garbage bin) parameters. All tunable via the GUI. --
GRIPPER_PID_ENABLE_VALUE = 10     # "PID3 10" enables PID mode on the gripper

# mechanical homing (reset) sequence - not exposed in the GUI (safety limits)
GRIPPER_HOMING_STEP_DEG      = 10
GRIPPER_HOMING_STEP_INTERVAL = 0.3
GRIPPER_HOMING_POLL_INTERVAL = 0.3
GRIPPER_HOMING_STABLE_TOL    = 1
GRIPPER_HOMING_STABLE_COUNT  = 3
GRIPPER_HOMING_MAX_ANGLE_MAG = 2000
GRIPPER_HOMING_TIMEOUT_SEC   = 30.0

# grip ramp (close then re-open) - startup defaults, all GUI-tunable via the
# "Gripper Tuning" panel: step size (deg), step time (s) and pause/hold
# time (s) - see MissionRunner.update_gripper_params().
GRIPPER_RAMP_MODE       = "linear"
GRIPPER_RAMP_MAX_ANGLE  = 330
GRIPPER_RAMP_STEP       = 50       # step size (deg)  - GUI tunable
GRIPPER_RAMP_INTERVAL   = 0.5      # step time (s)     - GUI tunable
GRIPPER_HOLD_AT_MAX_SEC = 1.0      # pause time (s)    - GUI tunable
GRIPPER_POST_WAIT_SEC   = 5.0

GRIPPER_ABORT_POLL_SEC  = 0.05

