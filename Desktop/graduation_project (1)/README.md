# Design and Control of Autonomous Waste-collection Vehicle

## 1. Project Overview

This project is a small autonomous Waste-Collection vehicle that:

1. **Follows a lane or a line** drawn on the floor using a camera and
   real-time computer vision, without any pre-built map.
2. **Detects an AprilTag** marker placed at a target/docking station while
   still driving, then **automatically stops and precisely centers itself**
   on the tag using a second camera stream and PID control.
3. **Operates a gripper** to pick up an object (e.g. a small bin) once
   centered on the tag.
4. Can be reset and repeat the whole cycle automatically, or be driven
   manually (W/A/S/D) with path recording and playback for testing.

The vehicle is controlled from a desktop Python/Tkinter application that
also acts as the "brain": it runs all computer vision and control logic on
a laptop/PC and sends motion commands to the vehicle's onboard controller
over Wi-Fi (TCP). The vehicle itself only executes low-level motor/servo
commands; no vision or decision-making happens on the vehicle.

## 2. Project Objectives

- Build a lane-following mobile robot.
- Recover gracefully from temporarily losing sight of the lane instead of
  stopping immediately, to tolerate real-world lighting/occlusion noise.
- Detect a fixed visual marker (AprilTag) to know precisely where to stop
  and dock, without relying on the lane lines for final positioning.
- Achieve millimeter-level centering accuracy on the marker using a
  dedicated PID loop before triggering the pickup mechanism.
- Automate a full pick-and-deliver cycle: drive → detect tag → center →
  grip → reset → repeat.
- Provide a single operator-friendly GUI to monitor, tune, and override
  every stage of the pipeline in real time (for testing/demo purposes).

## 3. System Architecture

```
                        ┌───────────────────────────────────────────┐
                        │              Desktop / Laptop              │
                        │        (Python + Tkinter application)      │
                        │                                             │
   PC Webcam  ────────▶ │  Lane detection & tracking  ──┐             │
   (lane view)          │  (OpenCV, STATE 1)            │             │
                        │                                ▼             │
                        │                        Lane-keeping PID      │
                        │                                │             │
   ESP32-CAM  ────────▶ │  AprilTag detection & pose    │             │
   (MJPEG, WiFi,        │  (STATE 2a: centering PID)     │             │
    port 8080)          │                                │             │
                        │  Mission state machine  ◀──────┘             │
                        │  (STATE 1→2→3→reset)                         │
                        │                │                              │
                        └────────────────┼──────────────────────────────┘
                                         │  TCP (port 4545)
                                         │  drive / steer / gripper cmds
                                         ▼
                        ┌───────────────────────────────────────────┐
                        │           Vehicle Controller Board          │
                        │   (drive motors + steering servo + gripper  │
                        │    motor, all with position/PID feedback)   │
                        └───────────────────────────────────────────┘
```

**Key idea:** all "intelligence" (vision + control loops + the state
machine) lives on the PC. The vehicle is a "dumb" actuator — an ESP32
running the firmware in [`firmware/`](firmware/) — that understands
a small text command protocol over TCP (see [How It Works](#7-how-it-works)).

## 4. Hardware Used

| Component | Role |
|---|---|
| Vehicle chassis | 
| ESP32 dev board (vehicle controller) | Runs `firmware/main/main.c` (ESP-IDF/FreeRTOS). Hosts the TCP command server (port 4545) that the PC app drives |
| 2× DC drive motors (standard dual-PWM driver) | Forward/reverse drive (**M1**, **M2** — GPIO32/33 and GPIO25/26 PWM, GPIO35/34 and GPIO36/39 encoders) |
| 1× DC motor + **L298N** driver | Gripper actuator (**M3** — ENA=GPIO2, IN1=GPIO0, IN2=GPIO15, encoder GPIO19/18) |
| 3× quadrature rotary encoders | Closed-loop position/speed feedback for M1/M2/M3 |
| 2× hobby servos | **SV1** = steering (GPIO13, range 1200–4000, center 2400); **SV2** = second servo, e.g. gripper claw or camera tilt (GPIO14) |
| **TF-Luna LiDAR** (UART) | Single-point distance sensor wired to UART2 (TX=GPIO17, RX=GPIO16, 115200 baud) — read on demand via the `LUNA` command |
| ESP32-CAM module | Streams MJPEG video (port 8080) used for AprilTag detection and stop-sign watching |
| PC / laptop webcam (or a second onboard camera) | Feeds the lane-following vision pipeline |
| Wi-Fi router / hotspot | Connects the PC, the vehicle controller, and the ESP32-CAM on the same network |
| Printed AprilTag marker (36h11 family recommended) | Docking/target marker |
| Black-and-yellow taped/drawn track | The lane the vehicle follows |

Full pin map and wiring notes: see [`firmware/README.md`](firmware/README.md#1-hardware--pin-map).

## 5. Software & Technologies

**PC application (`src/`)**
- **Python 3** — main application language
- **Tkinter / ttk** — desktop GUI (dark themed, tabbed control panels)
- **OpenCV (`opencv-python`)** — image processing: grayscale/HSV
  segmentation, morphology, connected components, Canny edge overlay,
  camera undistortion
- **NumPy** — polynomial curve fitting for lane boundaries, array math
- **Pillow (PIL)** — converting OpenCV frames to Tkinter-displayable images
- **pupil_apriltags** — AprilTag detection and pose estimation
- **Sockets (`socket`, TCP/UDP)** — communication with the vehicle
  controller (port 4545) and ESP32-CAM resolution control (UDP)
- **`threading` / `queue`** — background camera capture loops, MJPEG
  stream reading, and the mission state-machine loop, all running
  concurrently with the GUI's main thread

**Vehicle firmware (`firmware/`)**
- **C** on **ESP-IDF** (FreeRTOS) — the controller board's entire
  runtime
- **LEDC** (ESP32 hardware PWM) — motor and servo PWM generation
- **PCNT** (`driver/pulse_cnt.h`) — hardware quadrature-encoder counting
- **`esp_timer`** (hardware timers, ISR-dispatched) — deterministic PID
  loop ticking, decoupled from the actual PID math task
- **UART driver** — TF-Luna LiDAR frame parsing
- **lwIP sockets** — the TCP command server (port 4545) and Wi-Fi STA
  networking with automatic reconnect/backoff


## 6. How It Works

### 6.1 Lane detection pipeline (per frame)

1. Convert the frame to grayscale (for black lines) and HSV (for the
   yellow divider) in parallel.
2. Gaussian blur + an exponential-moving-average temporal filter to
   remove frame-to-frame flicker/noise.
3. Crop to a Region Of Interest (a horizontal band whose top edge is
   adjustable live from the GUI) so background clutter above the track
   is ignored.
4. Segment:
   - **Black lines**: inverse binary threshold on the filtered grayscale
     band.
   - **Yellow divider**: HSV `inRange` band-pass isolating the painted
     yellow line from the white floor/grey clutter.
   Both masks are cleaned with morphology and filtered by connected
   components (minimum area + elongation) to reject small objects lying
   on the track.
5. Sample N horizontal rows inside the ROI against both masks, turning
   colored pixel "runs" of a plausible width into candidate line points.
6. Chain points bottom-up into per-color tracks across consecutive rows,
   then fit a polynomial to each track so the boundary position can be
   predicted at the control row (even through curves or brief occlusion).

### 6.2 Lane topology, memory & color-aware lane typing

- Sorted boundaries become **lanes** (gaps of a plausible, learned width).
  Each lane is classified as `black_yellow` (yellow divider on one side)
  or `2_blacks` (no divider).
- On startup, the app asks the operator to confirm/shift the current
  lane; the first confirmed frame **calibrates** lane width, external
  space, and (for `black_yellow` lanes) which side the yellow line sits
  on relative to the black edge.
- In a `black_yellow` lane, the vehicle **follows the yellow line itself**
  (not the midpoint) — losing the yellow line is the critical failure,
  not the black edge.
- Boundaries that briefly leave the frame are kept alive in a short-term
  memory (position decayed/shifted with the visible ones) and can be
  **re-synthesized** from the learned lane geometry, so a temporary loss
  doesn't require stopping.

### 6.3 Mission flow (high level)

```
STATE 1  LANE_FOLLOWING  ──(AprilTag confirmed)──▶  TAG_TRANSITION (fixed pause)
                          ──(stop sign, stub)────▶  STOPSIGN_HALT ──▶ back to STATE 1
TAG_TRANSITION  ──▶  STATE 2a  CENTERING  (PID docking on the AprilTag)
STATE 2a  ──(locked/centered)──▶  STATE 3  GRIPPER  (grab sequence)
STATE 3  ──▶  AWAITING_RESET  ──(operator confirms)──▶  RESET  ──▶  STATE 1
```

The vehicle can be driven manually (W/A/S/D) at any time, with path
**recording** (`R`) and **playback** (`P`) for repeatable testing, and lane
keeping can be toggled independently with `L`.

## 7. State Machine Description

The mission runner (`src/mission/mission_runner.py`) drives a state
machine over `MissionState`:

| State | Meaning |
|---|---|
| `IDLE` | App just started, STATE 1 not running yet |
| `LANE_FOLLOWING` | STATE 1: lane tracker drives the vehicle; in parallel, the app watches the ESP32-CAM stream for an AprilTag and the PC camera for a stop sign |
| `TAG_TRANSITION` | Bridge state: tag confirmed, `STOP` sent, a fixed pause before centering starts |
| `CENTERING` | STATE 2a: PID-centers the vehicle on the AprilTag |
| `STOPSIGN_HALT` | STATE 2b: momentary stop for a detected stop sign, then lane following resumes automatically *(stop-sign detector is a stub — see `mission/mission_helpers.py::detect_stop_sign`)* |
| `GRIPPER` | STATE 3: runs the gripper homing → ramp-up → hold → ramp-down sequence |
| `AWAITING_RESET` | Cycle finished; waits for the operator to confirm before sending `RESET` and repeating |
| `ERROR` | Fault state |

A dedicated `GripperPhase` sub-state machine (`ENABLE_PID → HOMING → RESET
→ RAMP_UP → HOLD_MAX → RAMP_DOWN → DONE`) governs the gripper sequence
itself, and can be cleanly aborted mid-sequence (`GripperAborted`) if
E-STOP or a manual override is triggered.

Separately, the **lane-keeping controller** has its own internal
tracking state per lane: `TRACKING` (both required boundaries visible,
normal PID) → `PREDICTING` (yellow line visible but black edge lost, mild
slow-down) → `RECOVERING` (critical boundary lost, steer back toward its
last known side, slow down) → full loss (steer toward remembered lane
anchor for a limited number of frames) → `STOP` as the final fallback.

## 8. PID Control

Two independent PID loops are used:

**1. Lane-keeping steering PID** (`src/control/pid.py`, driven from
`src/control/lane_keep_controller.py`)
- Input: normalized pixel error between the steering **target** (lane
  midpoint, or the yellow-line position for `black_yellow` lanes, with a
  GUI-adjustable lateral offset) and the camera view's center.
- A first-image geometry-calibration correction is blended in if the
  vehicle drifts (lane width shrinking on one side while external space
  grows on the other).
- Anti wind-up via clamped + conditional integration, with integral reset
  inside the deadband.
- Below the error margin (deadband), the servo is commanded straight
  (`SERVO1 SET 2400`).
- Output is **slew-rate limited**: the steering value can change by at
  most 160 units per 50 ms to respect the servo's mechanical limits.

**2. AprilTag centering PID** (`src/control/center_pid.py`, driven from
`src/mission/mission_runner.py`)
- Input: millimetre offset from the tag's estimated pose (via
  `cv2.solvePnP`).
- Uses a **hysteretic (Schmitt-trigger) deadband**: once inside the
  margin, the error must exceed `margin × exit_ratio` again before the
  controller re-engages — this specifically fixes a derivative-kick /
  limit-cycle bug where the vehicle would settle and then immediately
  lurch again (see the patch notes preserved in
  `src/mission/mission_runner.py`).
- Output (mm) is converted to a wheel-rotation angle
  (`mm_to_wheel_degrees`) and sent as an absolute **position** command
  (`SET1`/`SET2`), not a velocity command — while inside the deadband, the
  controller **holds** the last commanded position instead of re-sending
  a misleading "0" command.
- Both Kp/Ki/Kd and the error margin are live-tunable from the GUI's
  "AprilTag PID" panel.

## 9. Installation / Setup

### 9.1 Vehicle controller firmware (one-time, per board)

Flash `firmware/` onto the vehicle's ESP32 with ESP-IDF:
```bash
cd firmware
idf.py set-target esp32
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```
Set your own Wi-Fi SSID/password in `firmware/main/main.c` first (see
[`firmware/README.md`](firmware/README.md#3-wi-fi-configuration)). Full
build details, pinout, and the TCP protocol reference live in that file.

### 9.2 PC application

**1. Clone/copy this repository** and make sure Python 3.9+ is installed.

**2. Install `tkinter`** (only needed on Linux; it ships with Python on
Windows/macOS):
```bash
sudo apt install python3-tk
```

**3. Install the Python dependencies:**
```bash
pip install -r requirements.txt
```

**4. Configure network constants** in `src/config.py` if your setup
differs from the defaults:
```python
ESP32_CAM_IP      = "192.168.137.123"   # your ESP32-CAM's IP
ESP32_CAM_PORT    = 8080
TCP_DEFAULT_PORT  = 4545                 # your vehicle controller's TCP port
```
The vehicle controller's IP is entered directly in the GUI at connect
time (it isn't hard-coded) — check the ESP-IDF serial monitor for the IP
it was assigned after connecting to Wi-Fi.

**5. Make sure the PC, the vehicle controller board, and the ESP32-CAM are
all on the same Wi-Fi network** before running the app.

## 10. How to Run

```bash
cd src
python main.py
```

1. Enter the vehicle controller's IP address in the GUI and click
   **CONNECT**.
2. Pick your lane-following camera from the camera dropdown (**DRIVE &
   PATH** tab) and start the preview.
3. Drive manually with **W / A / S / D**, or press **L** to toggle
   autonomous lane keeping once a lane has been confirmed.
4. The mission sidebar (STATE 2/3) starts automatically and watches the
   ESP32-CAM stream in the background — no separate step needed.
5. Tune detection, PID, and gripper parameters live from the tabs on the
   right; changes apply immediately.

**Keyboard shortcuts**

| Key | Action |
|---|---|
| `W` / `S` | Forward / reverse |
| `A` / `D` | Steer left / right |
| `R` | Start/stop path recording |
| `P` | Play back the last recorded path |
| `L` | Toggle autonomous lane keeping |

## 11. Team Members

| Name |

| Youssef Reda |
| Ibrahim Hassan | 
| Ahmed Medhat |
| Safi Eldeen Mohamed |
| Rahaf Maged |
| Nadeen Youssef|
| Mahmoud Said |
| Mohamed Shehata |
| Nermeen Ramdan |
| Ahmed Mohamed|


**Supervisor1:Prof. Abdelfatah Mahmoud Mohamed** 
**Supervisor2:Dr. Abdelrahman Ahmed Ali Morsi**
**Assiut University /Faculty of Engineering/ Electrical Department** 

## 12. Demo Video

> [▶ Watch the demo](https://drive.google.com/file/d/1tDQN-6lFjSLiaZXRuNxT87oG8NgTRr9t/view?usp=drive_link)

## 13. Screenshots

> ![GUI main screen](docs/images/gui_main_screen.png)
> ![Lane detection debug view](docs/images/lane_detection_debug_view.png)
> ![AprilTag centering](docs/images/apriltag_centering.png)
> ![Vehicle hardware](docs/images/vehicle_hardware.jpg)



## 14. Attachments

| File | Link |
|---|---|
| 📊 Presentation (PowerPoint) | [presentation.pptx](docs/Graduation Project Presentation/presentation.pptx) |
| 📄 Project Book | [Graduation_Project_Book.pdf](docs/project_book/Graduation_Project_Book.pdf) |
| 🖼️ Screenshots / Photos | [docs/images](docs/images/) |
| 🎥 Demo Video | [demo_video](docs/docs/demo/demo_video.mp4/demo_video.mp4)


## 15. Future Improvements

- Move vision/control processing onto an onboard embedded computer
  (e.g. Jetson Nano/Orin, Raspberry Pi) to remove the dependency on a
  tethered PC and the associated Wi-Fi latency.
- Add active lane-change execution (the topology model already exposes
  `request_lane_change()` and a `CHANGING_LANE` state hook — only the
  triggering logic and a lane-change trajectory are missing).
- Replace the hand-tuned HSV/threshold-based segmentation with a small
  learned segmentation model for better robustness to lighting changes.
- Add authentication/encryption on the TCP command channel before any
  deployment outside a controlled lab network.
- Log full mission runs (frames + commands + PID traces) automatically
  for offline tuning and regression testing.
- Add unit tests around `LaneTopologyModel`'s memory/recovery logic and
  the PID controllers, which are currently only exercised manually.

---

## Project Structure

```
.
├── README.md
├── requirements.txt
├── docs/
│   ├── presentation/     
│   ├── project_book/     
│   ├── images/           
│   └── demo/             
├── firmware/                          # ESP32 controller board firmware (ESP-IDF/C)
│   ├── README.md                      # build/flash steps, pinout, TCP protocol reference
│   ├── CMakeLists.txt
│   └── main/
│       ├── CMakeLists.txt
│       └── main.c                     # triple-motor PID + servo + TCP command server
└── src/                                # PC desktop application (Python)
    ├── main.py                        # entry point
    ├── config.py                      # all tunable constants
    ├── camera_utils.py                # local webcam discovery/open helpers
    ├── network/
    │   └── tcp_client.py              # shared TCP link to the vehicle (port 4545)
    ├── vision/
    │   ├── lane_detector.py           # per-frame black/yellow line segmentation
    │   ├── lane_topology.py           # lane pairing, memory, calibration
    │   ├── mjpeg_reader.py            # ESP32-CAM MJPEG stream reader
    │   ├── camera_resolution.py       # UDP stream-resolution control
    │   └── apriltag_utils.py          # camera geometry, pose, CLAHE/denoise
    ├── control/
    │   ├── pid.py                     # generic PID (lane-keeping steering)
    │   ├── center_pid.py              # AprilTag centering PID (hysteretic deadband)
    │   ├── lane_keep_controller.py    # STATE 1 autonomous driving loop
    │   └── vehicle_controller.py      # manual drive + record/playback host
    ├── mission/
    │   ├── mission_enums.py           # MissionState / GripperPhase / GripperAborted
    │   ├── mission_helpers.py         # small parsing/conversion helpers
    │   └── mission_runner.py          # full mission state machine (STATE 1-3)
    └── gui/
        └── app.py                     # Tkinter GUI (all tabs/panels)
```
