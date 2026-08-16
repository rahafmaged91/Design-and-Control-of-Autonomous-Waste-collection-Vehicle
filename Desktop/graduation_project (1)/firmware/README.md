# Vehicle Controller Firmware (ESP32 / ESP-IDF)

This is the firmware that runs **on the vehicle itself**. It is a
triple-motor PID controller built with **ESP-IDF** (FreeRTOS) that:

- Drives two standard dual-PWM motors (**M1**, **M2** — the drive wheels)
  and one **L298N-driven** motor (**M3**, e.g. the gripper).
- Drives two hobby servos (**SV1** = steering, **SV2** = a second servo,
  e.g. gripper claw / camera tilt) with optional auto-sweep.
- Reads three quadrature encoders (position + speed feedback for
  closed-loop PID).
- Reads a **TF-Luna LiDAR** distance sensor over UART2.
- Exposes all of the above over a plain-text **TCP command protocol on
  port 4545**, which is exactly what `src/network/tcp_client.py` in the
  PC application talks to.

This firmware is the counterpart to the Python desktop application in
`../src/` — see the top-level `README.md` for how the two fit together.

## 1. Hardware / pin map

| Signal | GPIO | Notes |
|---|---|---|
| M1 PWM A / B | 32 / 33 | Standard dual-PWM drive motor |
| M1 Encoder A / B | 35 / 34 | Quadrature |
| M2 PWM A / B | 25 / 26 | Standard dual-PWM drive motor |
| M2 Encoder A / B | 36 / 39 | Quadrature |
| M3 ENA (PWM/speed) | 2 | L298N-driven motor (e.g. gripper) |
| M3 IN1 / IN2 (direction) | 0 / 15 | ⚠️ GPIO0 is a boot-strapping pin on most ESP32 boards — often wired to the physical BOOT button. Double-check your board before relying on it as an output. |
| M3 Encoder A / B | 19 / 18 | Quadrature |
| SV1 (steering servo) | 13 | Range 1200–4000 (duty units), center 2400 |
| SV2 (second servo) | 14 | Range 1150–8600 (duty units) |
| TF-Luna LiDAR TX / RX | 17 / 16 | UART2, 115200 baud |

> SV1/SV2 were deliberately moved off GPIO3/GPIO1 (UART0 RXD/TXD) to
> GPIO13/GPIO14 — driving PWM on the UART0 pins collides with the
> serial console used for flashing/logging.

## 2. Build & flash (ESP-IDF)

Requires [ESP-IDF](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/) v5.x+ (this firmware uses the newer `driver/pulse_cnt.h`
PCNT API).

```bash
cd firmware
idf.py set-target esp32
idf.py menuconfig      # optional: adjust partition table / log level etc.
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor   # adjust the serial port for your OS
```

## 3. Wi-Fi configuration

The Wi-Fi SSID/password and TCP port are compile-time constants near the
top of `main/main.c`:

```c
#define WIFI_SSID        "Silver"  //put "YOUR_WIFI_SSID"
#define WIFI_PASS        "DoomSlayer11" // "YOUR_WIFI_PASSWORD"
#define TCP_PORT         4545
```



The board connects in STA (client) mode. On disconnect, it automatically
retries with a 2-second backoff (no manual reboot required after a Wi-Fi
drop). Once connected, watch the serial monitor for the assigned IP —
that's the address you enter in the PC app's **CONNECT** field.

## 4. TCP command protocol (port 4545)

Every command is a newline-terminated ASCII string; the board replies
with a single line starting `OK ...` or `ERR ...` (except `STOP`, which
replies `SLIME`, and unsolicited data isn't sent otherwise). `N` below
means `1`, `2`, or `3` (motor number); servo commands use `1` or `2`.

| Command | Effect |
|---|---|
| `STOP` | Immediately zero all 3 motors' PWM and stop all PID loops |
| `RESET` / `RESETN` | Zero the encoder count for all motors / motor N |
| `FLIPN` | Reverse motor N's direction (immediate, no reinit) |
| `FLIPNE` | Swap motor N's encoder A/B pins (requires re-init to take effect) |
| `PIDN <dt_ms>` | Start **position-mode** PID on motor N with a `dt_ms` update period |
| `PIDWN <dt_ms>` | Start **speed-mode** PID on motor N |
| `DIPN` | Stop PID (position or speed mode) on motor N |
| `RPMN <value>` | Set target RPM for motor N (speed mode only) |
| `REVN <revolutions>` | Set target position for motor N in wheel revolutions (position mode) |
| `SETN <angle_deg>` | While PID is running: set target position in **degrees** |
| `SETN P/I/D/E <value>` | While PID is running: tune Kp / Ki / Kd / error-margin live |
| `SETN <pwm> [dir]` | While PID is **not** running: open-loop PWM. For M3 (L298N), `dir` is `0`=coast, `1`=fwd, `2`=rev |
| `SETN <seconds>` (decimal) | Timed open-loop pulse: run for `seconds`, then auto-stop (non-blocking) |
| `PPRN <value>` | Set motor N's pulses-per-revolution (encoder resolution) |
| `SETALPHAN <0..1>` | Set the low-pass filter alpha for motor N's speed estimate |
| `ENCN?` | Read motor N's raw encoder count + revolutions |
| `POSN?` | Same as `ENCN?`, phrased as position |
| `SPEEDN?` | Read motor N's current filtered speed (RPM), speed mode only |
| `SERVOn SET <duty>` | Move servo n to an absolute duty value |
| `SERVOn MIN/MAX <duty>` | Set servo n's clamp range |
| `SERVOn CENTER` | Move servo n to the midpoint of its current MIN/MAX range |
| `SERVOn SWEEP` | Toggle auto-sweep on/off for servo n |
| `SERVOn STEP <duty>` | Set the auto-sweep step size for servo n |
| `SWPT <ms>` | Set the auto-sweep tick interval (shared by both servos) |
| `LUNA` | Read the last cached TF-Luna distance/strength/temperature frame |

This matches the `SET1`/`SET2`/`SERVO1 SET`/`STOP` commands referenced in
the PC app's protocol docstring — see the top-level `README.md`.

## 5. Architecture notes

- **PWM Gatekeeper pattern**: a single FreeRTOS task
  (`pwm_gatekeeper_task`, pinned to core 0) is the *only* code that ever
  calls `ledc_set_duty()` / `gpio_set_level()`. Every other task
  (including the 3 independent PID tasks) only ever enqueues a
  `pwm_cmd_t` onto a queue — this avoids concurrent hardware writes from
  multiple cores/tasks racing on the same LEDC channel.
- **PID timing**: each motor's PID loop is driven by its own
  `esp_timer` (hardware timer, ISR-dispatched) which only calls
  `vTaskNotifyGiveFromISR()` — the actual PID math runs in a normal
  FreeRTOS task (`pid_task`, pinned to core 1), never inside the ISR.
- **Graceful reconnection**: both the TCP `accept()` loop (1 s socket
  timeout) and the Wi-Fi `STA_DISCONNECTED` handler (2 s backoff timer)
  are designed to recover from a dropped connection or a Wi-Fi bounce
  without requiring a manual reboot.
- See the version-history comment block at the top of `main/main.c` for
  the detailed changelog of fixes (servo boot-centering, GPIO
  reassignment away from the UART console pins, the L298N direction-code
  validation, the `FLIP1`/`FLIP2` no-op bug fix, and the `FLIP3`
  cross-task race fix).
