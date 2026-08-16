/*
 * Triple Motor PID Controller – ESP-IDF v7  (M3 + L298N Support)
 *
 * ================================================================
 * M4 changelog (this revision) — requested fixes
 * ================================================================
 *  M4.1  Servo center-start fix.
 *          servo_init() previously programmed the LEDC channel with
 *          duty = duty_min but told the tracking state (duty_cur)
 *          it was at a hardcoded 4500 — a value that isn't even
 *          inside SV1's own clamp range (1400-3000). Both servos now
 *          boot to the true mechanical center of their configured
 *          min/max range, and duty_cur is set to match what the
 *          hardware is actually outputting.
 *
 *  M4.2  Servo GPIO reassignment.
 *          SV1/SV2 were wired to GPIO3 / GPIO1 — the default UART0
 *          RXD/TXD pins used by the console and flashing. Driving
 *          LEDC PWM on those pins collides with the serial console.
 *          Moved to GPIO13 / GPIO14, which were unused by anything
 *          else in this firmware and are not boot-strapping pins.
 *
 *  M4.3  TCP accept() timeout for graceful reconnection.
 *          accept() previously blocked forever with no timeout, so
 *          if WiFi dropped while blocked in accept(), the outer loop
 *          had no way to notice and rebuild the listening socket —
 *          the device could be stuck refusing all connections after
 *          a WiFi bounce until a full reboot. server_fd now has a 1s
 *          SO_RCVTIMEO so the WiFi-connected check runs at least once
 *          per second even with no incoming clients.
 *
 *  M4.4  WiFi reconnect backoff.
 *          WIFI_EVENT_STA_DISCONNECTED used to call esp_wifi_connect()
 *          synchronously and immediately, which can hammer the radio
 *          and the system event task on repeated drops. Reconnects
 *          are now scheduled via a one-shot esp_timer with a 2s
 *          backoff instead of firing inline.
 *
 *  M4.5  SET/timed-SET direction validation for L298N (M3).
 *          For M3, the second SET argument is a direction code
 *          (0=coast,1=fwd,2=rev), not a PWM value like it is for
 *          M1/M2. Previously an out-of-range value silently coasted
 *          the motor with no error, and the timed-pulse variant
 *          ("SET3 1.5") *always* passed dir=0, so M3 could never
 *          actually turn during a timed SET. Both are now handled
 *          correctly for is_l298 motors.
 *
 *  M4.6  FLIP1 / FLIP2 actually work now.
 *          These used to swap pwm_gpio_a/pwm_gpio_b, fields the
 *          gatekeeper never reads (it uses ledc_ch_a/ledc_ch_b, bound
 *          to physical pins once at boot) — so the command was a
 *          silent no-op despite replying "OK". They now swap
 *          ledc_ch_a/ledc_ch_b directly, which reverses direction
 *          immediately with no reinit required.
 *
 *  M4.7  pwm_cmd_t now carries the LEDC channel to drive.
 *          Removes the hardcoded LEDC_CHANNEL_4 in the L298N
 *          gatekeeper path and the M1/M2-only pointer lookup by idx.
 *          Adding a second L298N motor no longer requires editing
 *          the gatekeeper.
 *
 *  M4.8  FLIP3 race fix.
 *          FLIP3 swaps M3.in1_gpio/in2_gpio under cmd_mutex, but
 *          pid_task reads those same fields on its own schedule
 *          without that mutex, so a FLIP3 landing mid-cycle could
 *          hand the gatekeeper one new pin and one stale pin for a
 *          single PWM cycle. Both the read (in set_pwm) and the
 *          write (in FLIP3) are now wrapped in a short spinlock
 *          critical section.
 *
 * ----------------------------------------------------------------
 * Original M3 changelog (unchanged from v6):
 *  M3.1  Added motor M3 wired to an L298N driver:
 *          ENA  → GPIO 2  (PWM speed, single LEDC channel)
 *          IN1  → GPIO 0  (direction A)
 *          IN2  → GPIO 15 (direction B)
 *          ENC_A→ GPIO 19
 *          ENC_B→ GPIO 18
 *        NOTE: GPIO0 is a boot-strapping pin on most ESP32 boards
 *        and is often wired to a physical BOOT button. Driving it as
 *        a push-pull output is workable but double check your board
 *        doesn't tie a button to that net before relying on it.
 *
 *  M3.2  Added `bool is_l298` flag to motor_t.
 *          false → original dual-PWM-channel behaviour (M1, M2).
 *          true  → single PWM channel (ENA) + two GPIO direction pins (IN1/IN2).
 *
 *  M3.3  Added `int in1_gpio, in2_gpio` fields to motor_t for L298N
 *        direction pins.  Unused (0) for standard motors.
 *
 *  M3.4  PWM Gatekeeper updated: if cmd.is_l298 is set it writes one
 *        LEDC duty (ledc_ch_a) and drives two plain GPIO outputs for
 *        direction instead of writing two LEDC channels.
 *
 *  M3.5  set_pwm() and apply_pwm_output() updated to embed the
 *        is_l298 flag into pwm_cmd_t so the gatekeeper knows which
 *        path to take without touching shared motor state.
 *
 *  M3.6  process_command() extended:
 *          ENC3?   – read M3 encoder
 *          PPR3    – set M3 PPR
 *          FLIP3   – for is_l298 motors flips IN1/IN2 GPIOs instead
 *                    of swapping PWM channels.
 *          PID3 / PIDW3 / RPM3 / DIP3 / SET3 / REV3 /
 *          RESET3 / SETALPHA3 / SPEED3?  all supported via the
 *          existing generic handlers (motor_of extended to '3').
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <errno.h>              /* Fix B2: needed for EAGAIN / EWOULDBLOCK */
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/timers.h"
#include "freertos/event_groups.h"
#include "freertos/semphr.h"
#include "freertos/queue.h"
#include "driver/ledc.h"
#include "driver/gpio.h"          /* M3.4 – needed for L298N direction GPIOs */
#include "driver/uart.h"
#include "driver/pulse_cnt.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "nvs_flash.h"
#include "lwip/sockets.h"
#include "lwip/netdb.h"

/* ================================================================
 * Configuration
 * ================================================================ */
#define MOTOR_PWM_FREQ   20000
#define MOTOR_PWM_RES    LEDC_TIMER_10_BIT
#define PWM_MAX          1023
#define PWM_MIN          80
#define INTEGRAL_LIMIT   5000.0f
#define RPM_MAX          96.0f

#define SERVO_PWM_FREQ   50
#define SERVO_PWM_RES    LEDC_TIMER_16_BIT
#define SERVO_PWM_MAX    65535u
#define SERVO_SWEEP_MS_DEFAULT 10

#define UART_LUNA        UART_NUM_2
#define UART_LUNA_BAUD   115200
#define LUNA_TX_PIN      17
#define LUNA_RX_PIN      16
#define LUNA_BUF_SIZE    256
#define LUNA_FRAME_LEN   9
#define LUNA_HEADER      0x59

#define WIFI_SSID        "Silver"
#define WIFI_PASS        "DoomSlayer11"
#define TCP_PORT         4545
#define CMD_BUF_SIZE     128
#define MAX_CLIENTS      4

/* M3.1 – L298N pin assignments for M3 */
#define M3_ENA_GPIO      2   /* PWM speed output (ENA) */
#define M3_IN1_GPIO      0   /* direction pin A — see boot-strap warning above */
#define M3_IN2_GPIO      15  /* direction pin B */
#define M3_ENC_A_GPIO    19
#define M3_ENC_B_GPIO    18

/* M4.4 – WiFi reconnect backoff delay */
#define WIFI_RECONNECT_BACKOFF_US   2000000ULL   /* 2 s */

/* M4.3 – how often accept() wakes up to recheck WiFi state */
#define TCP_ACCEPT_TIMEOUT_SEC      1

static const char *TAG = "MOTOR_CTRL";

/* ================================================================
 * Motor struct
 * ================================================================ */
typedef struct {
    int              pwm_gpio_a, pwm_gpio_b;  /* standard dual-PWM pins */
    int              enc_gpio_a,  enc_gpio_b;
    ledc_channel_t   ledc_ch_a,   ledc_ch_b;
    pcnt_unit_handle_t pcnt_unit;

    /* M3.2 – L298N flag ------------------------------------------ */
    bool             is_l298;   /* false = dual-PWM (M1/M2), true = L298N (M3) */

    /* M3.3 – L298N direction GPIOs (unused / 0 for M1, M2) -------- */
    int              in1_gpio;  /* L298N IN1 pin */
    int              in2_gpio;  /* L298N IN2 pin */

    float  Kp, Ki, Kd, delta_t;

    int64_t target_position, encoder_offset, current_count;
    int     error_margin, last_error;

    bool    speed_mode;
    float   target_rpm, target_diff;
    float   filtered_speed, last_error_speed;
    float   lp_alpha;

    float   integral;
    int     PPR;

    int64_t enc_buff[11];
    int     speed_idx, speed_samples;

    esp_timer_handle_t esp_timer;

    bool             pid_running;
    TaskHandle_t     task_handle;
    bool             target_reached;

    TimerHandle_t    pwm_off_timer;
} motor_t;

static motor_t M1, M2, M3;  /* M3.1 – third motor instance */

/* M4.8 – spinlock protecting M3.in1_gpio/in2_gpio against the
 * FLIP3 (process_command context) vs set_pwm (pid_task context) race. */
static portMUX_TYPE m3_dir_lock = portMUX_INITIALIZER_UNLOCKED;

/* ================================================================
 * Servo struct & instances
 * ================================================================ */
typedef struct {
    int            pin;
    ledc_channel_t ledc_ch;
    ledc_timer_t   ledc_timer;
    uint32_t       duty_min;
    uint32_t       duty_max;
    uint32_t       duty_cur;
    uint32_t       sweep_step;
    bool           sweep_en;
    int            sweep_dir;
} servo_t;
static servo_t SV1, SV2;

/* ================================================================
 * PWM Gatekeeper: single task owns all LEDC/GPIO hardware
 * ================================================================ */
typedef enum { PWM_MOTOR, PWM_SERVO } pwm_type_t;

typedef struct {
    pwm_type_t type;
    int        idx;        /* 0=M1/SV1, 1=M2/SV2, 2=M3 (used for logging only) */
    uint32_t   duty_a;
    uint32_t   duty_b;     /* for standard motors: ch_b duty; for L298N: direction code */
    /* M3.5 – embed L298N flag so gatekeeper needs no motor pointer */
    bool       is_l298;    /* copy of motor_t.is_l298 */
    int        in1_gpio;   /* copy of motor_t.in1_gpio (L298N only) */
    int        in2_gpio;   /* copy of motor_t.in2_gpio (L298N only) */
    /* M4.7 – carry the LEDC channel(s) to drive so the gatekeeper
     * never needs to hardcode a channel or dereference &M1/&M2. */
    ledc_channel_t ledc_ch_a;
    ledc_channel_t ledc_ch_b;  /* unused for L298N */
} pwm_cmd_t;

static QueueHandle_t pwm_queue      = NULL;
static TaskHandle_t  pwm_task_handle = NULL;

/* ----------------------------------------------------------------
 * PWM Gatekeeper task
 * M3.4 – extended to handle L298N single-channel + GPIO direction
 * M4.7 – channel now comes from the command, not a hardcoded value
 * ---------------------------------------------------------------- */
static void pwm_gatekeeper_task(void *arg)
{
    pwm_cmd_t cmd;
    while (1) {
        if (xQueueReceive(pwm_queue, &cmd, portMAX_DELAY) != pdTRUE) continue;

        if (cmd.type == PWM_MOTOR) {

            /* ---- M3.4: L298N path -------------------------------- */
            if (cmd.is_l298) {
                /*
                 * duty_a encodes the magnitude (0..PWM_MAX).
                 * duty_b encodes direction:
                 *   1 → forward  (IN1=1, IN2=0)
                 *   2 → reverse  (IN1=0, IN2=1)
                 *   0 → coast    (IN1=0, IN2=0)
                 */
                uint32_t speed = cmd.duty_a;
                if (speed > (uint32_t)PWM_MAX) speed = (uint32_t)PWM_MAX;

                /* Set direction GPIOs before enabling PWM */
                if (cmd.duty_b == 1) {           /* forward */
                    gpio_set_level(cmd.in1_gpio, 1);
                    gpio_set_level(cmd.in2_gpio, 0);
                } else if (cmd.duty_b == 2) {    /* reverse */
                    gpio_set_level(cmd.in1_gpio, 0);
                    gpio_set_level(cmd.in2_gpio, 1);
                } else {                         /* coast / stop */
                    gpio_set_level(cmd.in1_gpio, 0);
                    gpio_set_level(cmd.in2_gpio, 0);
                }

                /* M4.7 – channel comes from the command now; adding a
                 * second L298N motor only means populating ledc_ch_a
                 * correctly in that motor's motor_t, no gatekeeper edit. */
                ledc_set_duty(LEDC_HIGH_SPEED_MODE, cmd.ledc_ch_a, speed);
                ledc_update_duty(LEDC_HIGH_SPEED_MODE, cmd.ledc_ch_a);

            /* ---- standard dual-PWM path (M1, M2) ----------------- */
            } else {
                uint32_t da = cmd.duty_a;
                uint32_t db = cmd.duty_b;
                if (da > (uint32_t)PWM_MAX) da = (uint32_t)PWM_MAX;
                if (db > (uint32_t)PWM_MAX) db = (uint32_t)PWM_MAX;
                ledc_set_duty(LEDC_HIGH_SPEED_MODE, cmd.ledc_ch_a, da);
                ledc_update_duty(LEDC_HIGH_SPEED_MODE, cmd.ledc_ch_a);
                ledc_set_duty(LEDC_HIGH_SPEED_MODE, cmd.ledc_ch_b, db);
                ledc_update_duty(LEDC_HIGH_SPEED_MODE, cmd.ledc_ch_b);
            }

        } else if (cmd.type == PWM_SERVO) {
            servo_t *s = (cmd.idx == 0) ? &SV1 : &SV2;
            uint32_t d = cmd.duty_a;
            if (d < s->duty_min) d = s->duty_min;
            if (d > s->duty_max) d = s->duty_max;
            s->duty_cur = d;
            ledc_set_duty(LEDC_LOW_SPEED_MODE, s->ledc_ch, d);
            ledc_update_duty(LEDC_LOW_SPEED_MODE, s->ledc_ch);
        }
    }
}

static void pwm_enqueue(const pwm_cmd_t *cmd)
{
    if (!pwm_queue) return;
    if (xQueueSend(pwm_queue, cmd, pdMS_TO_TICKS(5)) != pdPASS) {
        ESP_LOGW(TAG, "PWM queue full – dropped %s%d",
                 cmd->type == PWM_MOTOR ? "M" : "S", cmd->idx);
    }
}

/* ================================================================
 * Sweep interval
 * ================================================================ */
static volatile uint32_t g_sweep_interval_ms = SERVO_SWEEP_MS_DEFAULT;

/* ================================================================
 * Shared-state protection
 * ================================================================ */
static SemaphoreHandle_t cmd_mutex;
static SemaphoreHandle_t client_sem;

/* ================================================================
 * TCP send helpers
 * ================================================================ */
static void tcp_send(int fd, const char *msg)
{
    if (fd < 0) return;
    send(fd, msg, strlen(msg), 0);
    send(fd, "\n", 1, 0);
}

static void tcp_sendf(int fd, const char *fmt, ...) __attribute__((format(printf,2,3)));
static void tcp_sendf(int fd, const char *fmt, ...)
{
    if (fd < 0) return;
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    tcp_send(fd, buf);
}

/* ================================================================
 * Servo helpers
 * ================================================================ */
static void servo_set_duty(servo_t *s, uint32_t duty)
{
    pwm_cmd_t cmd = {
        .type   = PWM_SERVO,
        .idx    = (s == &SV1) ? 0 : 1,
        .duty_a = duty,
        .duty_b = 0,
    };
    pwm_enqueue(&cmd);
}

static void sweep_task(void *arg)
{
    (void)arg;
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(g_sweep_interval_ms));
        servo_t *sv[2] = { &SV1, &SV2 };
        for (int i = 0; i < 2; i++) {
            servo_t *s = sv[i];
            if (!s->sweep_en || s->sweep_step == 0) continue;
            int32_t next = (int32_t)s->duty_cur
                         + (int32_t)(s->sweep_dir * (int32_t)s->sweep_step);
            if (next >= (int32_t)s->duty_max) { next = (int32_t)s->duty_max; s->sweep_dir = -1; }
            else if (next <= (int32_t)s->duty_min) { next = (int32_t)s->duty_min; s->sweep_dir = +1; }
            s->duty_cur = (uint32_t)next;
            servo_set_duty(s, s->duty_cur);
        }
    }
}

/*
 * servo_init – M4.1
 * Boots the servo to the mechanical CENTER of its configured
 * min/max range, and keeps the tracked duty_cur consistent with what
 * the hardware is actually outputting. Previously this programmed
 * duty_min into the LEDC channel but told duty_cur it was at a
 * hardcoded 4500 — a value outside SV1's own clamp range — so
 * software believed the servo was somewhere it could never reach,
 * and the physical servo silently sat at one extreme instead of
 * center on boot.
 */
static void servo_init(servo_t *s)
{
    uint32_t center = 2400;

    ledc_channel_config_t ch = {
        .channel    = s->ledc_ch, .duty      = center,
        .gpio_num   = s->pin,     .speed_mode = LEDC_LOW_SPEED_MODE,
        .timer_sel  = s->ledc_timer, .hpoint  = 0,
    };
    ESP_ERROR_CHECK(ledc_channel_config(&ch));
    s->duty_cur  = center;
    s->sweep_dir = +1;
}

/* ================================================================
 * Encoder helpers
 * ================================================================ */
static int64_t count_update(motor_t *m)
{
    int raw = 0;
    pcnt_unit_get_count(m->pcnt_unit, &raw);
    m->current_count += (int64_t)(int16_t)raw;
    pcnt_unit_clear_count(m->pcnt_unit);
    return m->current_count;
}

static int64_t get_encoder_count(motor_t *m)
{
    int raw = 0;
    pcnt_unit_get_count(m->pcnt_unit, &raw);
    return m->encoder_offset + (int16_t)raw;
}

static void reset_encoder(motor_t *m)
{
    pcnt_unit_clear_count(m->pcnt_unit);
    m->encoder_offset = 0;
    m->current_count  = 0;
}

/* ================================================================
 * Omega filter (float)
 * ================================================================ */
static float get_omega_filtered(int64_t *buf, int size)
{
    int n = size - 1;
    float diff[10];
    for (int i = 0; i < n; i++)
        diff[i] = (float)(buf[i + 1] - buf[i]);
    for (int i = 1; i < n; i++) {
        float key = diff[i]; int j = i - 1;
        while (j >= 0 && diff[j] > key) { diff[j+1] = diff[j]; j--; }
        diff[j+1] = key;
    }
    float sum = 0.0f;
    for (int i = 1; i < n - 1; i++) sum += diff[i];
    return sum / (float)(n - 2);
}

/* ================================================================
 * Motor PWM helpers
 * M3.5 – set_pwm() and apply_pwm_output() embed is_l298 flag
 * M4.7 – set_pwm() also carries the LEDC channel(s) to drive
 * M4.8 – set_pwm() reads M3's in1_gpio/in2_gpio under a spinlock
 * ================================================================ */
static void set_pwm(motor_t *m, uint32_t da, uint32_t db)
{
    if (da > (uint32_t)PWM_MAX) da = (uint32_t)PWM_MAX;
    if (db > (uint32_t)PWM_MAX) db = (uint32_t)PWM_MAX;

    pwm_cmd_t cmd = {
        .type      = PWM_MOTOR,
        .idx       = (m == &M1) ? 0 : (m == &M2) ? 1 : 2,   /* M3.5 */
        .duty_a    = da,
        .duty_b    = db,
        /* M3.5 – copy L298N metadata into the command */
        .is_l298   = m->is_l298,
        /* M4.7 – copy channel metadata into the command */
        .ledc_ch_a = m->ledc_ch_a,
        .ledc_ch_b = m->ledc_ch_b,
    };

    /* M4.8 – in1_gpio/in2_gpio can be swapped concurrently by FLIP3
     * running in another task; read them together as a pair under a
     * short critical section so the gatekeeper never sees a torn mix
     * of one new pin and one stale pin. */
    portENTER_CRITICAL(&m3_dir_lock);
    cmd.in1_gpio = m->in1_gpio;
    cmd.in2_gpio = m->in2_gpio;
    portEXIT_CRITICAL(&m3_dir_lock);

    pwm_enqueue(&cmd);
}

/*
 * apply_pwm_output – M3.5
 * For standard motors: da = |output| forward, db = |output| reverse (as before).
 * For L298N motors:    da = |output| (speed), db = direction code (1/2/0).
 */
static void apply_pwm_output(motor_t *m, float output)
{
    int val = (int)fabsf(output);
    if (val < PWM_MIN) val = PWM_MIN;
    if (val > PWM_MAX) val = PWM_MAX;

    if (m->is_l298) {
        /* M3.5: encode direction into duty_b field */
        if      (output > 0.0f) set_pwm(m, (uint32_t)val, 1); /* forward */
        else if (output < 0.0f) set_pwm(m, (uint32_t)val, 2); /* reverse */
        else                    set_pwm(m, 0, 0);              /* coast   */
    } else {
        /* original dual-channel behaviour */
        if      (output > 0.0f) set_pwm(m, (uint32_t)val, 0);
        else if (output < 0.0f) set_pwm(m, 0, (uint32_t)val);
        else                    set_pwm(m, 0, 0);
    }
}

/* ================================================================
 * One-shot software timer callback (timed SET)
 * ================================================================ */
static void pwm_off_timer_cb(TimerHandle_t xTimer)
{
    motor_t *m = (motor_t *)pvTimerGetTimerID(xTimer);
    set_pwm(m, 0, 0);
    ESP_LOGI(TAG, "Timed PWM off: motor%s",
             (m == &M1) ? "1" : (m == &M2) ? "2" : "3");
}

/* ================================================================
 * PID task (Core 1, notified by esp_timer ISR)
 * ================================================================ */
static void pid_task(void *arg)
{
    motor_t *m = (motor_t *)arg;
    while (1) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        if (!m->pid_running) { m->current_count = 0; m->speed_idx = 0; continue; }

        int64_t total = count_update(m);

        if (m->speed_mode) {
            m->enc_buff[m->speed_idx] = total;
            m->speed_idx = (m->speed_idx + 1) % 11;
            if (m->speed_samples < 11) m->speed_samples++;
            if (m->speed_samples < 11) { set_pwm(m, 0, 0); continue; }

            float omega = get_omega_filtered(m->enc_buff, 11);
            m->filtered_speed = (1.0f - m->lp_alpha) * m->filtered_speed
                              + m->lp_alpha * omega;

            float err = m->target_diff - m->filtered_speed;
            m->integral += (err + m->last_error_speed) * 0.5f * m->delta_t;
            if (m->integral >  INTEGRAL_LIMIT) m->integral =  INTEGRAL_LIMIT;
            if (m->integral < -INTEGRAL_LIMIT) m->integral = -INTEGRAL_LIMIT;
            float deriv = (err - m->last_error_speed) / m->delta_t;
            m->last_error_speed = err;
            apply_pwm_output(m, m->Kp * err + m->Ki * m->integral + m->Kd * deriv);

        } else {
            int64_t err = m->target_position - total;
            if (llabs(err) < m->error_margin) {
                m->integral = 0; m->last_error = 0;
                set_pwm(m, 0, 0); m->target_reached = true;
                continue;
            }
            m->integral += (float)(err + m->last_error) * 0.5f * m->delta_t;
            if (m->integral >  INTEGRAL_LIMIT) m->integral =  INTEGRAL_LIMIT;
            if (m->integral < -INTEGRAL_LIMIT) m->integral = -INTEGRAL_LIMIT;
            float deriv   = (float)(err - m->last_error) / m->delta_t;
            m->last_error = (int)err;
            apply_pwm_output(m, m->Kp * (float)err + m->Ki * m->integral + m->Kd * deriv);
        }
    }
}

/* ================================================================
 * esp_timer ISR callback (only notifies – NEVER touches LEDC/GPIO)
 * ================================================================ */
static void IRAM_ATTR pid_esp_timer_cb(void *arg)
{
    motor_t *m = (motor_t *)arg;
    BaseType_t woken = pdFALSE;
    vTaskNotifyGiveFromISR(m->task_handle, &woken);
    if (woken) portYIELD_FROM_ISR();
}

/* ================================================================
 * Timer lifecycle helpers
 * ================================================================ */
static void timer_init_if_needed(motor_t *m)
{
    if (m->esp_timer != NULL) return;
    const char *name = (m == &M1) ? "M1 PID Timer"
                     : (m == &M2) ? "M2 PID Timer"
                     :              "M3 PID Timer";  /* M3.1 */
    esp_timer_create_args_t args = {
        .callback              = pid_esp_timer_cb,
        .arg                   = (void *)m,
        .name                  = name,
        .skip_unhandled_events = true,
#if CONFIG_ESP_TIMER_SUPPORTS_ISR_DISPATCH_METHOD
        .dispatch_method = ESP_TIMER_ISR,
#else
        .dispatch_method = ESP_TIMER_TASK,

#endif
    };
    ESP_ERROR_CHECK(esp_timer_create(&args, &m->esp_timer));
}

static void timer_start(motor_t *m, int dt_ms)
{
    if (dt_ms <= 0) { ESP_LOGW(TAG, "Invalid delta_t"); return; }
    if (m->pid_running) {
        ESP_ERROR_CHECK(esp_timer_stop(m->esp_timer));
        m->pid_running = false;
    }
    ESP_ERROR_CHECK(esp_timer_start_periodic(m->esp_timer, (uint64_t)dt_ms * 1000ULL));
    m->pid_running = true;
}

static void timer_stop(motor_t *m)
{
    if (m->esp_timer != NULL && m->pid_running) esp_timer_stop(m->esp_timer);
    m->pid_running      = false;
    m->integral         = 0.0f;
    m->last_error       = 0;
    m->last_error_speed = 0.0f;
    m->filtered_speed   = 0.0f;
    set_pwm(m, 0, 0);
}

/* ================================================================
 * Encoder init (shared by all three motors)
 * ================================================================ */
static void encoder_init(motor_t *m)
{
    pcnt_unit_config_t ucfg = { .low_limit = -32768, .high_limit = 32767 };
    ESP_ERROR_CHECK(pcnt_new_unit(&ucfg, &m->pcnt_unit));

    pcnt_chan_config_t ca = { .edge_gpio_num = m->enc_gpio_a, .level_gpio_num = m->enc_gpio_b };
    pcnt_channel_handle_t hca = NULL;
    ESP_ERROR_CHECK(pcnt_new_channel(m->pcnt_unit, &ca, &hca));
    ESP_ERROR_CHECK(pcnt_channel_set_edge_action(hca, PCNT_CHANNEL_EDGE_ACTION_INCREASE, PCNT_CHANNEL_EDGE_ACTION_DECREASE));
    ESP_ERROR_CHECK(pcnt_channel_set_level_action(hca, PCNT_CHANNEL_LEVEL_ACTION_KEEP,    PCNT_CHANNEL_LEVEL_ACTION_INVERSE));

    pcnt_chan_config_t cb = { .edge_gpio_num = m->enc_gpio_b, .level_gpio_num = m->enc_gpio_a };
    pcnt_channel_handle_t hcb = NULL;
    ESP_ERROR_CHECK(pcnt_new_channel(m->pcnt_unit, &cb, &hcb));
    ESP_ERROR_CHECK(pcnt_channel_set_edge_action(hcb, PCNT_CHANNEL_EDGE_ACTION_INCREASE, PCNT_CHANNEL_EDGE_ACTION_DECREASE));
    ESP_ERROR_CHECK(pcnt_channel_set_level_action(hcb, PCNT_CHANNEL_LEVEL_ACTION_INVERSE, PCNT_CHANNEL_LEVEL_ACTION_KEEP));

    ESP_ERROR_CHECK(pcnt_unit_enable(m->pcnt_unit));
    ESP_ERROR_CHECK(pcnt_unit_clear_count(m->pcnt_unit));
    ESP_ERROR_CHECK(pcnt_unit_start(m->pcnt_unit));
}

/* ================================================================
 * M3.1 – L298N direction GPIO init
 * Call this once after encoder_init(&M3) in app_main.
 * ================================================================ */
static void l298n_gpio_init(motor_t *m)
{
    /* Configure IN1 and IN2 as push-pull outputs, initially low (coast) */
    gpio_config_t io = {
        .pin_bit_mask = (1ULL << m->in1_gpio) | (1ULL << m->in2_gpio),
        .mode         = GPIO_MODE_OUTPUT,
        .pull_up_en   = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&io));
    gpio_set_level(m->in1_gpio, 0);
    gpio_set_level(m->in2_gpio, 0);
}

/* ================================================================
 * TF-Luna LiDAR
 * ================================================================ */
typedef struct { uint16_t dist_cm; uint16_t strength; float temp_c; } luna_frame_t;
static volatile luna_frame_t luna_cache       = {0};
static volatile bool         luna_cache_valid = false;

static void luna_uart_init(void)
{
    uart_config_t cfg = {
        .baud_rate = UART_LUNA_BAUD, .data_bits = UART_DATA_8_BITS,
        .parity    = UART_PARITY_DISABLE, .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE, .source_clk = UART_SCLK_APB,
    };
    ESP_ERROR_CHECK(uart_driver_install(UART_LUNA, LUNA_BUF_SIZE, 0, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_param_config(UART_LUNA, &cfg));
    ESP_ERROR_CHECK(uart_set_pin(UART_LUNA, LUNA_TX_PIN, LUNA_RX_PIN,
                                 UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
}

static bool luna_parse_frame(const uint8_t *buf, int len, luna_frame_t *out)
{
    for (int i = 0; i <= len - LUNA_FRAME_LEN; i++) {
        if (buf[i] != LUNA_HEADER || buf[i+1] != LUNA_HEADER) continue;
        uint8_t cksum = 0;
        for (int j = 0; j < 8; j++) cksum += buf[i+j];
        if (cksum != buf[i+8]) continue;
        out->dist_cm  = (uint16_t)(buf[i+2] | ((uint16_t)buf[i+3] << 8));
        out->strength = (uint16_t)(buf[i+4] | ((uint16_t)buf[i+5] << 8));
        uint16_t rt   = (uint16_t)(buf[i+6] | ((uint16_t)buf[i+7] << 8));
        out->temp_c   = (rt / 8.0f) - 256.0f;
        return true;
    }
    return false;
}

static void luna_task(void *arg)
{
    (void)arg;
    uint8_t buf[LUNA_BUF_SIZE];
    luna_frame_t f;
    while (1) {
        int len = uart_read_bytes(UART_LUNA, buf, sizeof(buf), pdMS_TO_TICKS(500));
        if (len > 0 && luna_parse_frame(buf, len, &f)) {
            luna_cache = f; luna_cache_valid = true;
        }
    }
}

/* ================================================================
 * WiFi (STA mode)
 * M4.4 – reconnects are now scheduled via a backoff timer instead of
 *        calling esp_wifi_connect() synchronously inside the event
 *        handler on every single disconnect.
 * ================================================================ */
#define WIFI_CONNECTED_BIT BIT0
static EventGroupHandle_t wifi_eg;
static esp_timer_handle_t wifi_reconnect_timer = NULL;

static void wifi_reconnect_timer_cb(void *arg)
{
    (void)arg;
    ESP_LOGI(TAG, "WiFi reconnect attempt...");
    esp_wifi_connect();
}

static void wifi_event_handler(void *arg, esp_event_base_t base,
                                int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        xEventGroupClearBits(wifi_eg, WIFI_CONNECTED_BIT);
        ESP_LOGW(TAG, "WiFi disconnected – reconnecting in %llu ms",
                 WIFI_RECONNECT_BACKOFF_US / 1000ULL);
        /* M4.4 – schedule the retry instead of calling esp_wifi_connect()
         * inline; ignore the "not running" error from stop(), it's fine
         * if no retry was already pending. */
        (void)esp_timer_stop(wifi_reconnect_timer);
        (void)esp_timer_start_once(wifi_reconnect_timer, WIFI_RECONNECT_BACKOFF_US);
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *e = (ip_event_got_ip_t *)data;
        ESP_LOGI(TAG, "WiFi IP: " IPSTR, IP2STR(&e->ip_info.ip));
        xEventGroupSetBits(wifi_eg, WIFI_CONNECTED_BIT);
    }
}

static void wifi_init(void)
{
    wifi_eg = xEventGroupCreate();
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    /* M4.4 – reconnect backoff timer, created once and reused for
     * every future disconnect via esp_timer_start_once(). */
    esp_timer_create_args_t rc_args = {
        .callback = wifi_reconnect_timer_cb,
        .arg      = NULL,
        .name     = "wifi_reconnect",
    };
    ESP_ERROR_CHECK(esp_timer_create(&rc_args, &wifi_reconnect_timer));

    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID,    wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT,   IP_EVENT_STA_GOT_IP, wifi_event_handler, NULL));
    wifi_config_t wcfg = { .sta = { .ssid = WIFI_SSID, .password = WIFI_PASS } };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wcfg));
    ESP_ERROR_CHECK(esp_wifi_start());
    esp_wifi_connect();
}

/* ================================================================
 * Command helpers
 * M3.6 – motor_of() extended to return &M3 for '3'
 * ================================================================ */
static motor_t *motor_of(char c)
{
    if (c == '1') return &M1;
    if (c == '2') return &M2;
    if (c == '3') return &M3;   /* M3.6 */
    return NULL;
}
static servo_t *servo_of(char c) { return (c == '1') ? &SV1 : &SV2; }

static void start_speed_mode(motor_t *m, int dt_ms, int fd)
{
    m->delta_t = dt_ms / 1000.0f;
    timer_stop(m);
    m->speed_mode = true; m->speed_samples = 0; m->speed_idx = 0;
    m->filtered_speed = 0.0f; m->last_error_speed = 0.0f;
    int64_t sc = count_update(m);
    for (int i = 0; i < 11; i++) m->enc_buff[i] = sc;
    timer_init_if_needed(m);
    timer_start(m, dt_ms);
    tcp_sendf(fd, "OK PIDW%c dt=%d ms started",
              (m == &M1) ? '1' : (m == &M2) ? '2' : '3', dt_ms);
}

/* ================================================================
 * Command processor
 * M3.6 – all motor-N commands now work for N = 1, 2, or 3.
 *         FLIP3 is L298N-aware (flips direction GPIOs, not PWM pins).
 * M4.5  – SET/timed-SET now handle L298N direction correctly.
 * M4.6  – FLIP1/FLIP2 actually reverse direction now.
 * M4.8  – FLIP3 swap is now protected by a spinlock.
 * ================================================================ */
void process_command(char *cmd, int fd)
{
    /* ---- STOP ---------------------------------------------------- */
    if (strcmp(cmd, "STOP") == 0) {
        set_pwm(&M1, 0, 0); set_pwm(&M2, 0, 0); set_pwm(&M3, 0, 0); /* M3.6 */
        timer_stop(&M1); timer_stop(&M2); timer_stop(&M3);
        M1.target_rpm = 0; M2.target_rpm = 0; M3.target_rpm = 0;
        tcp_send(fd, "SLIME");

    /* ---- RESET ---------------------------------------------------- */
    } else if (strcmp(cmd, "RESET") == 0) {
        reset_encoder(&M1); reset_encoder(&M2); reset_encoder(&M3); /* M3.6 */
        tcp_send(fd, "OK encoders reset");
    } else if (strcmp(cmd, "RESET1") == 0) { reset_encoder(&M1); tcp_send(fd, "OK encoder1 reset");
    } else if (strcmp(cmd, "RESET2") == 0) { reset_encoder(&M2); tcp_send(fd, "OK encoder2 reset");
    } else if (strcmp(cmd, "RESET3") == 0) { reset_encoder(&M3); tcp_send(fd, "OK encoder3 reset"); /* M3.6 */

    /* ---- FLIP – M3.6, fixed in M4.6/M4.8 -------------------------- */
    } else if (strcmp(cmd, "FLIP1") == 0) {
        /* M4.6: swap the LEDC *channels* (what the gatekeeper actually
         * reads) instead of the unused pwm_gpio_a/b fields. Reverses
         * direction immediately, no reinit needed. */
        ledc_channel_t t = M1.ledc_ch_a; M1.ledc_ch_a = M1.ledc_ch_b; M1.ledc_ch_b = t;
        tcp_send(fd, "OK FLIP1: direction reversed (immediate)");
    } else if (strcmp(cmd, "FLIP2") == 0) {
        ledc_channel_t t = M2.ledc_ch_a; M2.ledc_ch_a = M2.ledc_ch_b; M2.ledc_ch_b = t;
        tcp_send(fd, "OK FLIP2: direction reversed (immediate)");

    /*
     * M3.6 – FLIP3: for an L298N motor there are no dual PWM channels
     * to swap.  Instead we swap IN1 and IN2 GPIO numbers so every
     * subsequent direction command will have its polarity reversed.
     * M4.8 – the swap is now wrapped in a spinlock shared with
     * set_pwm()'s read of the same fields, closing the cross-task race.
     */
    } else if (strcmp(cmd, "FLIP3") == 0) {
        if (M3.is_l298) {
            portENTER_CRITICAL(&m3_dir_lock);
            int t = M3.in1_gpio; M3.in1_gpio = M3.in2_gpio; M3.in2_gpio = t;
            portEXIT_CRITICAL(&m3_dir_lock);
            tcp_send(fd, "OK FLIP3 L298N: IN1/IN2 swapped (immediate)");
        } else {
            /* Fallback: treat like a standard motor if flag ever changes */
            int t = M3.pwm_gpio_a; M3.pwm_gpio_a = M3.pwm_gpio_b; M3.pwm_gpio_b = t;
            tcp_send(fd, "OK FLIP3 (reinit PWM to take effect)");
        }

    } else if (strcmp(cmd, "FLIP1E") == 0) {
        int t = M1.enc_gpio_a; M1.enc_gpio_a = M1.enc_gpio_b; M1.enc_gpio_b = t;
        tcp_send(fd, "OK FLIP1E (reinit encoder to take effect)");
    } else if (strcmp(cmd, "FLIP2E") == 0) {
        int t = M2.enc_gpio_a; M2.enc_gpio_a = M2.enc_gpio_b; M2.enc_gpio_b = t;
        tcp_send(fd, "OK FLIP2E (reinit encoder to take effect)");
    } else if (strcmp(cmd, "FLIP3E") == 0) { /* M3.6 */
        int t = M3.enc_gpio_a; M3.enc_gpio_a = M3.enc_gpio_b; M3.enc_gpio_b = t;
        tcp_send(fd, "OK FLIP3E (reinit encoder to take effect)");

    /* ---- PIDW (speed mode start) --------------------------------- */
    } else if (strncmp(cmd, "PIDW1", 5) == 0 || strncmp(cmd, "PIDW2", 5) == 0
                                              || strncmp(cmd, "PIDW3", 5) == 0) { /* M3.6 */
        motor_t *m = motor_of(cmd[4]);
        char *tok = strtok(cmd + 6, " ");
        if (tok) start_speed_mode(m, atoi(tok), fd);
        else tcp_send(fd, "ERR PIDW: missing dt");

    /* ---- RPM setpoint -------------------------------------------- */
    } else if (strncmp(cmd, "RPM1", 4) == 0 || strncmp(cmd, "RPM2", 4) == 0
                                             || strncmp(cmd, "RPM3", 4) == 0) { /* M3.6 */
        motor_t *m = motor_of(cmd[3]);
        if (!m->speed_mode) {
            tcp_sendf(fd, "ERR RPM%c: start PIDW first", cmd[3]);
        } else {
            char *tok = strtok(cmd + 5, " ");
            if (tok) {
                float rpm = atof(tok);
                if (rpm >  RPM_MAX) rpm =  RPM_MAX;
                if (rpm < -RPM_MAX) rpm = -RPM_MAX;
                m->target_rpm  = rpm;
                m->target_diff = rpm * (float)m->PPR * m->delta_t / 60.0f;
                tcp_sendf(fd, "OK RPM%c=%.2f target_diff=%.4f", cmd[3], rpm, m->target_diff);
            } else tcp_sendf(fd, "ERR RPM%c: missing value", cmd[3]);
        }

    /* ---- DIP (stop PID) ------------------------------------------ */
    } else if (strcmp(cmd, "DIP1") == 0 || strcmp(cmd, "DIP2") == 0
                                        || strcmp(cmd, "DIP3") == 0) { /* M3.6 */
        motor_t *m = motor_of(cmd[3]);
        if (m->pwm_off_timer) xTimerStop(m->pwm_off_timer, 0);
        m->speed_mode = false; m->speed_samples = 0;
        timer_stop(m);
        tcp_sendf(fd, "OK PID%c stopped", cmd[3]);

    /* ---- PID (position mode start) ------------------------------- */
    } else if ((strncmp(cmd, "PID1", 4) == 0 || strncmp(cmd, "PID2", 4) == 0
                                              || strncmp(cmd, "PID3", 4) == 0) /* M3.6 */
               && (cmd[4] == '\0' || cmd[4] == ' ')) {
        motor_t *m = motor_of(cmd[3]);
        char *tok = strtok(cmd + 5, " ");
        if (tok) {
            int dt = atoi(tok);
            m->delta_t = dt / 1000.0f;
            timer_init_if_needed(m);
            timer_start(m, dt);
            tcp_sendf(fd, "OK PID%c started dt=%d ms", cmd[3], dt);
        } else tcp_sendf(fd, "ERR PID%c: missing dt", cmd[3]);

    /* ---- PPR ----------------------------------------------------- */
    } else if (strncmp(cmd, "PPR1", 4) == 0 || strncmp(cmd, "PPR2", 4) == 0
                                             || strncmp(cmd, "PPR3", 4) == 0) { /* M3.6 */
        motor_t *m = motor_of(cmd[3]);
        char *tok = strtok(cmd + 5, " ");
        if (tok) { m->PPR = atoi(tok); tcp_sendf(fd, "OK PPR%c=%d", cmd[3], m->PPR); }
        else tcp_sendf(fd, "ERR PPR%c: missing value", cmd[3]);

    /* ---- SPEED? -------------------------------------------------- */
    } else if (strcmp(cmd, "SPEED1?") == 0 || strcmp(cmd, "SPEED2?") == 0
                                           || strcmp(cmd, "SPEED3?") == 0) { /* M3.6 */
        motor_t *m = motor_of(cmd[5]);
        if (!m->speed_mode) {
            tcp_sendf(fd, "ERR SPEED%c?: start PIDW first", cmd[5]);
        } else {
            float rpm = (m->filtered_speed * 60.0f) / ((float)m->PPR * m->delta_t);
            tcp_sendf(fd, "OK SPEED%c=%.2f RPM", cmd[5], rpm);
        }

    /* ---- POS? ---------------------------------------------------- */
    } else if (strcmp(cmd, "POS1?") == 0 || strcmp(cmd, "POS2?") == 0
                                         || strcmp(cmd, "POS3?") == 0) { /* M3.6 */
        motor_t *m = motor_of(cmd[3]);
        int64_t c = get_encoder_count(m);
        tcp_sendf(fd, "OK POS%c=%lld counts (%.3f rev)", cmd[3], c, (float)c / (float)m->PPR);

    /* ---- REV ----------------------------------------------------- */
    } else if (strncmp(cmd, "REV1", 4) == 0 || strncmp(cmd, "REV2", 4) == 0
                                             || strncmp(cmd, "REV3", 4) == 0) { /* M3.6 */
        motor_t *m = motor_of(cmd[3]);
        char *tok = strtok(cmd + 5, " ");
        if (tok) {
            float r = atof(tok);
            m->target_position = (int64_t)(r * (float)m->PPR);
            tcp_sendf(fd, "OK REV%c=%.2f → %lld counts", cmd[3], r, m->target_position);
        } else tcp_sendf(fd, "ERR REV%c: missing value", cmd[3]);

    /* ---- SET – M4.5: L298N direction now validated ---------------- */
    } else if (strncmp(cmd, "SET1", 4) == 0 || strncmp(cmd, "SET2", 4) == 0
                                             || strncmp(cmd, "SET3", 4) == 0) { /* M3.6 */
        motor_t *m = motor_of(cmd[3]);
        char *tok = strtok(cmd + 5, " ");
        if (!tok) {
            tcp_sendf(fd, "ERR SET%c: no argument", cmd[3]);
        } else if (m->pid_running) {
            if      (strcmp(tok,"P")==0){char *v=strtok(NULL," ");if(v){m->Kp=atof(v);tcp_sendf(fd,"OK Kp%c=%.3f",cmd[3],m->Kp);}else tcp_sendf(fd,"ERR missing value");}
            else if (strcmp(tok,"I")==0){char *v=strtok(NULL," ");if(v){m->Ki=atof(v);tcp_sendf(fd,"OK Ki%c=%.3f",cmd[3],m->Ki);}else tcp_sendf(fd,"ERR missing value");}
            else if (strcmp(tok,"D")==0){char *v=strtok(NULL," ");if(v){m->Kd=atof(v);tcp_sendf(fd,"OK Kd%c=%.3f",cmd[3],m->Kd);}else tcp_sendf(fd,"ERR missing value");}
            else if (strcmp(tok,"E")==0){char *v=strtok(NULL," ");if(v){m->error_margin=atoi(v);tcp_sendf(fd,"OK err_margin%c=%d",cmd[3],m->error_margin);}else tcp_sendf(fd,"ERR missing value");}
            else {
                float angle = atof(tok);
                m->target_position = (int64_t)((angle / 360.0f) * (float)m->PPR);
                tcp_sendf(fd, "OK SET%c %.2f° → %lld counts", cmd[3], angle, m->target_position);
            }
        } else {
            if (strchr(tok, '.') != NULL) {
                float secs = atof(tok);
                if (secs > 0.0f) {
                    /*
                     * M4.5: for an L298N motor duty_b is a direction
                     * code, not a PWM value. This previously always
                     * passed 0 (coast), so a timed SET on M3 spun the
                     * ENA line with both direction pins low and the
                     * motor never actually turned. Default the timed
                     * pulse to "forward" for L298N motors, matching
                     * the forward-channel-only behaviour already used
                     * for standard dual-PWM motors.
                     */
                    set_pwm(m, PWM_MIN, m->is_l298 ? 1 : 0);
                    if (m->pwm_off_timer == NULL) {
                        m->pwm_off_timer = xTimerCreate(
                            "pwm_off", pdMS_TO_TICKS(100),
                            pdFALSE, (void *)m, pwm_off_timer_cb);
                    }
                    xTimerChangePeriod(m->pwm_off_timer,
                                       pdMS_TO_TICKS((uint32_t)(secs * 1000.0f)), 0);
                    xTimerStart(m->pwm_off_timer, 0);
                    tcp_sendf(fd, "OK SET%c timed %.2f s (non-blocking)", cmd[3], secs);
                } else tcp_sendf(fd, "ERR SET%c: duration must be > 0", cmd[3]);
            } else {
                uint32_t da = (uint32_t)atoi(tok);
                char *t2 = strtok(NULL, " ");
                if (m->is_l298) {
                    /*
                     * M4.5: validate the direction code instead of
                     * silently coasting when a caller passes a raw
                     * PWM-style second argument copy-pasted from
                     * SET1/SET2 usage (which expects a duty value,
                     * not 0/1/2).
                     */
                    uint32_t dir = t2 ? (uint32_t)atoi(t2) : 0;
                    if (dir > 2) {
                        tcp_sendf(fd, "ERR SET%c: dir must be 0=coast 1=fwd 2=rev", cmd[3]);
                    } else {
                        set_pwm(m, da, dir);
                        tcp_sendf(fd, "OK SET%c PWM=%lu dir=%lu", cmd[3],
                                  (unsigned long)da, (unsigned long)dir);
                    }
                } else {
                    set_pwm(m, da, t2 ? (uint32_t)atoi(t2) : 0);
                    tcp_sendf(fd, "OK SET%c PWM=%ld/%ld", cmd[3], da, t2 ? (uint32_t)atoi(t2) : 0u);
                }
            }
        }

    /* ---- ENC? – M3.6 now handles '3' too ------------------------- */
    } else if (strcmp(cmd, "ENC1?") == 0 || strcmp(cmd, "ENC2?") == 0
                                         || strcmp(cmd, "ENC3?") == 0) { /* M3.6 */
        motor_t *m = motor_of(cmd[3]);
        int64_t c = get_encoder_count(m);
        tcp_sendf(fd, "OK ENC%c=%lld (%.3f rev)", cmd[3], c, (float)c / (float)m->PPR);

    /* ---- SETALPHA ------------------------------------------------ */
    } else if (strncmp(cmd, "SETALPHA1", 9) == 0 || strncmp(cmd, "SETALPHA2", 9) == 0
                                                  || strncmp(cmd, "SETALPHA3", 9) == 0) { /* M3.6 */
        motor_t *m = motor_of(cmd[8]);
        char *tok = strtok(cmd + 10, " ");
        if (tok) {
            float a = atof(tok);
            if (a < 0.0f) a = 0.0f;
            if (a > 1.0f) a = 1.0f;
            m->lp_alpha = a;
            tcp_sendf(fd, "OK alpha%c=%.3f β=%.3f", cmd[8], a, 1.0f - a);
        } else tcp_sendf(fd, "ERR SETALPHA%c: missing value", cmd[8]);

    /* ---- SERVO --------------------------------------------------- */
    } else if (strncmp(cmd, "SERVO1", 6) == 0 || strncmp(cmd, "SERVO2", 6) == 0) {
        servo_t *s = servo_of(cmd[5]);
        char *sub = strtok(cmd + 7, " ");
        if (!sub) {
            tcp_sendf(fd, "ERR SERVO%c: no subcommand", cmd[5]);
        } else if (strcmp(sub,"MIN")==0){
            char *v=strtok(NULL," ");
            if(v){uint32_t d=(uint32_t)atoi(v);if(d>SERVO_PWM_MAX)d=SERVO_PWM_MAX;s->duty_min=d;tcp_sendf(fd,"OK SERVO%c min=%ld",cmd[5],d);}
            else tcp_sendf(fd,"ERR SERVO%c MIN: missing value",cmd[5]);
        } else if (strcmp(sub,"MAX")==0){
            char *v=strtok(NULL," ");
            if(v){uint32_t d=(uint32_t)atoi(v);if(d>SERVO_PWM_MAX)d=SERVO_PWM_MAX;s->duty_max=d;tcp_sendf(fd,"OK SERVO%c max=%ld",cmd[5],d);}
            else tcp_sendf(fd,"ERR SERVO%c MAX: missing value",cmd[5]);
        } else if (strcmp(sub,"SET")==0){
            char *v=strtok(NULL," ");
            if(v){servo_set_duty(s,(uint32_t)atoi(v));tcp_sendf(fd,"OK SERVO%c→%ld",cmd[5],s->duty_cur);}
            else tcp_sendf(fd,"ERR SERVO%c SET: missing value",cmd[5]);
        } else if (strcmp(sub,"SWEEP")==0){
            s->sweep_en=!s->sweep_en;
            tcp_sendf(fd,"OK SERVO%c sweep %s",cmd[5],s->sweep_en?"ON":"OFF");
        } else if (strcmp(sub,"STEP")==0){
            char *v=strtok(NULL," ");
            if(v){s->sweep_step=(uint32_t)atoi(v);tcp_sendf(fd,"OK SERVO%c step=%ld",cmd[5],s->sweep_step);}
            else tcp_sendf(fd,"ERR SERVO%c STEP: missing value",cmd[5]);
        } else if (strcmp(sub,"CENTER")==0){
            /* M4.1 – explicit re-center command, handy after MIN/MAX changes */
            uint32_t center = s->duty_min + (s->duty_max - s->duty_min) / 2;
            servo_set_duty(s, center);
            tcp_sendf(fd,"OK SERVO%c centered=%ld",cmd[5],center);
        } else {
            tcp_sendf(fd, "ERR SERVO%c unknown sub: %s", cmd[5], sub);
        }

    /* ---- SWPT ---------------------------------------------------- */
    } else if (strncmp(cmd, "SWPT", 4) == 0) {
        char *tok = strtok(cmd + 5, " ");
        if (!tok) {
            tcp_sendf(fd, "ERR SWPT: missing ms  (current=%" PRIu32 " ms)", g_sweep_interval_ms);
        } else {
            int ms = atoi(tok);
            if (ms < 1) ms = 1;
            g_sweep_interval_ms = (uint32_t)ms;
            tcp_sendf(fd, "OK sweep interval=%d ms", ms);
        }

    /* ---- LUNA ---------------------------------------------------- */
    } else if (strcmp(cmd, "LUNA") == 0) {
        if (!luna_cache_valid)
            tcp_send(fd, "ERR LUNA: no frame received yet");
        else
            tcp_sendf(fd, "OK LUNA dist=%u cm  strength=%u  temp=%.2f C",
                      luna_cache.dist_cm, luna_cache.strength, luna_cache.temp_c);

    } else {
        tcp_sendf(fd, "ERR unknown command: %s", cmd);
        ESP_LOGW(TAG, "Unknown cmd: %s", cmd);
    }
}

/* ================================================================
 * Per-client handler task
 * ================================================================ */
typedef struct { int fd; } client_arg_t;

static void client_handler_task(void *arg)
{
    client_arg_t *ca = (client_arg_t *)arg;
    int fd = ca->fd;
    free(ca);
    char line[CMD_BUF_SIZE];
    int  line_idx = 0;
    char byte;
    tcp_send(fd, "READY motor_ctrl_v7");
    while (1) {
        int n = recv(fd, &byte, 1, 0);
        if (n == 0) break;  /* Fix 2: clean FIN from client */
        if (n < 0) {
            /* Fix 2: SO_RCVTIMEO fires EAGAIN – treat as dead connection */
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                ESP_LOGW(TAG, "Client recv timeout – closing connection");
            } else {
                ESP_LOGW(TAG, "Client recv error %d – closing connection", errno);
            }
            break;
        }
        if (byte == '\r') continue;
        if (byte == '\n') {
            line[line_idx] = '\0';
            if (line_idx > 0) {
                /* Fix 1: copy into a separate buffer so strtok() inside
                 * process_command() cannot corrupt the live recv line[]. */
                char cmd_copy[CMD_BUF_SIZE];
                memcpy(cmd_copy, line, line_idx + 1);
                xSemaphoreTake(cmd_mutex, portMAX_DELAY);
                process_command(cmd_copy, fd);
                xSemaphoreGive(cmd_mutex);
            }
            line_idx = 0;
        } else if (line_idx < CMD_BUF_SIZE - 1) {
            line[line_idx++] = byte;
        }
    }
    /* Fix 4: stop all motors when a client disconnects so nothing
     * keeps running with no one able to send STOP. */
    timer_stop(&M1); timer_stop(&M2); timer_stop(&M3);
    set_pwm(&M1, 0, 0); set_pwm(&M2, 0, 0); set_pwm(&M3, 0, 0);
    /* Ensure the socket is fully torn down before releasing the slot,
     * so a fast client reconnect never races a lingering fd. */
    shutdown(fd, SHUT_RDWR);
    close(fd);
    ESP_LOGI(TAG, "Client disconnected – all motors stopped");
    xSemaphoreGive(client_sem);
    vTaskDelete(NULL);
}

/* ================================================================
 * TCP server task
 * M4.3 – server_fd now has an accept() timeout so a WiFi drop is
 *        detected within ~1s and the socket is gracefully rebuilt
 *        on reconnect every time, instead of potentially blocking
 *        forever in accept() with no way to notice the interface
 *        went down.
 * ================================================================ */
static void tcp_server_task(void *arg)
{
    (void)arg;

    /*
     * Fix 5: outer loop – re-create the server socket from scratch every
     * time WiFi reconnects.  Previously the socket was created once and
     * became permanently invalid after the first WiFi drop.
     */
    while (1) {

        /* Wait (or re-wait after a drop) until we have an IP address */
        xEventGroupWaitBits(wifi_eg, WIFI_CONNECTED_BIT, pdFALSE, pdTRUE, portMAX_DELAY);
        ESP_LOGI(TAG, "WiFi up – starting TCP server on :%d", TCP_PORT);

        int server_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (server_fd < 0) {
            ESP_LOGE(TAG, "socket() failed – retrying in 1 s");
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }

        int reuse = 1;
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

        /* M4.3 – give accept() a timeout so the WiFi-connected check
         * below runs at least once per second even with no incoming
         * clients, instead of blocking indefinitely. */
        struct timeval accept_timeout = { .tv_sec = TCP_ACCEPT_TIMEOUT_SEC, .tv_usec = 0 };
        setsockopt(server_fd, SOL_SOCKET, SO_RCVTIMEO, &accept_timeout, sizeof(accept_timeout));

        struct sockaddr_in addr = {
            .sin_family      = AF_INET,
            .sin_port        = htons(TCP_PORT),
            .sin_addr.s_addr = htonl(INADDR_ANY),
        };
        if (bind(server_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
            ESP_LOGE(TAG, "bind() failed – retrying in 1 s");
            close(server_fd);
            vTaskDelay(pdMS_TO_TICKS(1000));
            continue;
        }
        listen(server_fd, MAX_CLIENTS);
        ESP_LOGI(TAG, "TCP listening on :%d (max %d clients)", TCP_PORT, MAX_CLIENTS);

        /*
         * Fix 5 / M4.3: inner accept loop – exits promptly when WiFi
         * drops (thanks to the accept() timeout above) so the outer
         * loop can tear down server_fd and re-bind after reconnection
         * every time, not just when a client happens to try to connect.
         */
        while (xEventGroupGetBits(wifi_eg) & WIFI_CONNECTED_BIT) {
            struct sockaddr_in client_addr;
            socklen_t addr_len = sizeof(client_addr);
            int client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &addr_len);
            if (client_fd < 0) {
                /* Either a real error or the 1s accept timeout — either
                 * way just loop back and recheck the WiFi bit. */
                vTaskDelay(pdMS_TO_TICKS(100));
                continue;
            }
            if (xSemaphoreTake(client_sem, 0) != pdTRUE) {
                send(client_fd, "BUSY max clients reached\n", 25, 0);
                close(client_fd);
                ESP_LOGW(TAG, "Rejected client: max connections reached");
                continue;
            }

            /* Fix 2: 10-second receive timeout so dead connections are
             * always detected and client_sem is always released (Fix 6). */
            struct timeval rx_timeout = { .tv_sec = 10, .tv_usec = 0 };
            setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &rx_timeout, sizeof(rx_timeout));

            /* Fix 3: TCP keepalive so half-open connections are detected
             * by the lwIP stack in addition to the application timeout. */
            int ka = 1;
            setsockopt(client_fd, SOL_SOCKET, SO_KEEPALIVE, &ka, sizeof(ka));

            client_arg_t *ca = malloc(sizeof(client_arg_t));
            if (!ca) { close(client_fd); xSemaphoreGive(client_sem); continue; }
            ca->fd = client_fd;
            ESP_LOGI(TAG, "Client connected");
            BaseType_t ret = xTaskCreatePinnedToCore(client_handler_task, "cli", 4096, ca, 4, NULL, 0);
            if (ret != pdPASS) { close(client_fd); free(ca); xSemaphoreGive(client_sem); }
        }

        /* WiFi dropped – close the server socket and go back to waiting */
        ESP_LOGW(TAG, "WiFi lost – closing server socket, will re-listen on reconnect");
        close(server_fd);
        vTaskDelay(pdMS_TO_TICKS(500));   /* brief pause before re-waiting */
    }
}

/* ================================================================
 * LEDC motor channel init helper (standard dual-PWM)
 * ================================================================ */
static void motor_ledc_ch_init(ledc_channel_t ch, int gpio)
{
    ledc_channel_config_t c = {
        .channel = ch, .duty = 0, .gpio_num = gpio,
        .speed_mode = LEDC_HIGH_SPEED_MODE, .timer_sel = LEDC_TIMER_0, .hpoint = 0,
    };
    ESP_ERROR_CHECK(ledc_channel_config(&c));
}

/* ================================================================
 * M3.1 – LEDC init for L298N ENA (single speed channel)
 * Uses HIGH_SPEED_MODE / TIMER_0 to share the same motor timer.
 * ================================================================ */
static void l298n_ledc_init(ledc_channel_t ch, int gpio)
{
    ledc_channel_config_t c = {
        .channel    = ch,
        .duty       = 0,
        .gpio_num   = gpio,
        .speed_mode = LEDC_HIGH_SPEED_MODE,
        .timer_sel  = LEDC_TIMER_0,   /* same 20 kHz timer as M1/M2 */
        .hpoint     = 0,
    };
    ESP_ERROR_CHECK(ledc_channel_config(&c));
}

/* ================================================================
 * app_main
 * ================================================================ */
void app_main(void)
{
    cmd_mutex  = xSemaphoreCreateMutex();
    client_sem = xSemaphoreCreateCounting(MAX_CLIENTS, MAX_CLIENTS);

    /* ---- M1 (standard dual-PWM) ---------------------------------- */
    M1 = (motor_t){
        .pwm_gpio_a = 32, .pwm_gpio_b = 33,
        .enc_gpio_a = 35, .enc_gpio_b = 34,
        .ledc_ch_a  = LEDC_CHANNEL_0, .ledc_ch_b = LEDC_CHANNEL_1,
        .is_l298    = false,           /* M3.2 – standard driver */
        .Kp = 10.0f, .Ki = 8.0f, .Kd = 1.0f,
        .delta_t = 0.01f, .error_margin = 10, .PPR = 4030, .lp_alpha = 0.2f,
    };

    /* ---- M2 (standard dual-PWM) ---------------------------------- */
    M2 = (motor_t){
        .pwm_gpio_a = 25, .pwm_gpio_b = 26,
        .enc_gpio_a = 36, .enc_gpio_b = 39,
        .ledc_ch_a  = LEDC_CHANNEL_2, .ledc_ch_b = LEDC_CHANNEL_3,
        .is_l298    = false,           /* M3.2 – standard driver */
        .Kp = 10.0f, .Ki = 8.0f, .Kd = 1.0f,
        .delta_t = 0.01f, .error_margin = 10, .PPR = 4030, .lp_alpha = 0.2f,
    };

    /*
     * M3.1 – M3 (L298N driver)
     * ENA  → GPIO 2  (PWM, LEDC_CHANNEL_4)
     * IN1  → GPIO 0  (direction A) — see boot-strap warning at top of file
     * IN2  → GPIO 15 (direction B)
     * ENC_A→ GPIO 19
     * ENC_B→ GPIO 18
     */
    M3 = (motor_t){
        .pwm_gpio_a = M3_ENA_GPIO,    /* ENA pin (speed PWM) */
        .pwm_gpio_b = 0,              /* unused for L298N */
        .enc_gpio_a = M3_ENC_A_GPIO,
        .enc_gpio_b = M3_ENC_B_GPIO,
        .ledc_ch_a  = LEDC_CHANNEL_4, /* ENA channel */
        .ledc_ch_b  = 0,              /* unused for L298N */
        /* M3.2 – flag this motor as L298N */
        .is_l298    = true,
        /* M3.3 – direction GPIO pins */
        .in1_gpio   = M3_IN1_GPIO,
        .in2_gpio   = M3_IN2_GPIO,
        .Kp = 2.0f, .Ki = 2.0f, .Kd = 1.0f,
        .delta_t = 0.01f, .error_margin = 10, .PPR = 1594, .lp_alpha = 0.2f,
    };

    /* ---- LEDC timer (20 kHz, 10-bit) shared by all motors -------- */
    ledc_timer_config_t mt = {
        .duty_resolution = MOTOR_PWM_RES, .freq_hz = MOTOR_PWM_FREQ,
        .speed_mode = LEDC_HIGH_SPEED_MODE, .timer_num = LEDC_TIMER_0,
        .clk_cfg = LEDC_AUTO_CLK,
    };
    ESP_ERROR_CHECK(ledc_timer_config(&mt));

    /* M1 / M2 standard channels */
    motor_ledc_ch_init(LEDC_CHANNEL_0, M1.pwm_gpio_a);
    motor_ledc_ch_init(LEDC_CHANNEL_1, M1.pwm_gpio_b);
    motor_ledc_ch_init(LEDC_CHANNEL_2, M2.pwm_gpio_a);
    motor_ledc_ch_init(LEDC_CHANNEL_3, M2.pwm_gpio_b);

    /* M3.1 – M3 ENA channel */
    l298n_ledc_init(LEDC_CHANNEL_4, M3_ENA_GPIO);

    /* ---- Servo LEDC (50 Hz, 16-bit) ------------------------------ */
    ledc_timer_config_t st = {
        .duty_resolution = SERVO_PWM_RES, .freq_hz = SERVO_PWM_FREQ,
        .speed_mode = LEDC_LOW_SPEED_MODE, .timer_num = LEDC_TIMER_1,
        .clk_cfg = LEDC_AUTO_CLK,
    };
    ESP_ERROR_CHECK(ledc_timer_config(&st));

    /*
     * M4.2 – SV1/SV2 moved off GPIO3/GPIO1 (UART0 RXD/TXD, used by the
     * console and flashing) onto GPIO13/GPIO14, which are free and not
     * boot-strapping pins.
     *
     * SV1 → LEDC_CHANNEL_5 (LOW_SPEED)
     * SV2 → LEDC_CHANNEL_6 (LOW_SPEED)
     * Independent of M3's LEDC_CHANNEL_4 (HIGH_SPEED) — no conflict,
     * numbered sequentially just to avoid confusion.
     */
    SV1 = (servo_t){ .pin=13, .ledc_ch=LEDC_CHANNEL_5, .ledc_timer=LEDC_TIMER_1,
                     .duty_min=1200, .duty_max=4000, .sweep_step=200, .sweep_dir=+1 };
    SV2 = (servo_t){ .pin=14, .ledc_ch=LEDC_CHANNEL_6, .ledc_timer=LEDC_TIMER_1,
                     .duty_min=1150, .duty_max=8600, .sweep_step=50,  .sweep_dir=+1 };
    /*
     * NOTE on duty_min/duty_max: at 50 Hz / 16-bit resolution each LEDC
     * count is ~0.305 us (20000 us / 65536). Standard hobby servos
     * expect ~500-2500 us pulses, i.e. roughly 1638-8192 counts. SV1's
     * current range (1200-4000, ~369-1229 us) sits below that window —
     * worth re-checking against your actual servo's datasheet if its
     * motion feels clipped at one end. Left unchanged here since this
     * may already be tuned for your specific hardware.
     */
    servo_init(&SV1);   /* M4.1 – now boots to true center, not duty_min */
    servo_init(&SV2);
    //forcing a center on boot is useful for servos that have been moved to an extreme position and need to be centered before the first command is sent. This prevents any potential damage or misalignment.
    servo_set_duty(&SV1, 2400);
    luna_uart_init();
    encoder_init(&M1);
    encoder_init(&M2);
    encoder_init(&M3);        /* M3.1 – init M3 encoder */
    l298n_gpio_init(&M3);     /* M3.1 – init L298N direction GPIOs */

    /* E1: create PWM queue and gatekeeper BEFORE any task can enqueue */
    pwm_queue = xQueueCreate(64, sizeof(pwm_cmd_t));
    xTaskCreatePinnedToCore(pwm_gatekeeper_task, "pwm_gk", 3072, NULL, 10, &pwm_task_handle, 0);

    /* Safe initial zero-PWM via gatekeeper */
    set_pwm(&M1, 0, 0);
    set_pwm(&M2, 0, 0);
    set_pwm(&M3, 0, 0);       /* M3.1 */

    wifi_init();

    /*
     * Task map
     * ─────────────────────────────────────────────────────────────
     * Core 1 (PID computation only – NEVER touches LEDC/GPIO)
     *   pri 10  pid1  – esp_timer ISR → ulTaskNotifyTake
     *   pri 10  pid2  – esp_timer ISR → ulTaskNotifyTake
     *   pri 10  pid3  – esp_timer ISR → ulTaskNotifyTake  [M3.1]
     *
     * Core 0 (hardware writers + I/O)
     *   pri 10  pwm_gk – sole owner of ledc_set_duty and IN1/IN2 GPIO
     *   pri  5  sweep  – vTaskDelay(g_sweep_interval_ms)
     *   pri  5  luna   – blocks on UART2 read
     *   pri  4  tcp    – accept() loop (1s timeout), spawns cli tasks
     *   pri  4  cli[]  – per-client recv() + process_command()
     */
    xTaskCreatePinnedToCore(pid_task,        "pid1",  4096, &M1, 10, &M1.task_handle, 1);
    xTaskCreatePinnedToCore(pid_task,        "pid2",  4096, &M2, 10, &M2.task_handle, 1);
    xTaskCreatePinnedToCore(pid_task,        "pid3",  4096, &M3, 10, &M3.task_handle, 1); /* M3.1 */
    xTaskCreatePinnedToCore(sweep_task,      "sweep", 2048, NULL, 5, NULL, 0);
    xTaskCreatePinnedToCore(luna_task,       "luna",  3072, NULL, 5, NULL, 0);
    xTaskCreatePinnedToCore(tcp_server_task, "tcp",   4096, NULL, 4, NULL, 0);

    ESP_LOGI(TAG, "v7 running – M1/M2 dual-PWM, M3 L298N, servos centered on boot,");
    ESP_LOGI(TAG, "graceful WiFi/TCP reconnection enabled");
    ESP_LOGI(TAG, "Connect to %s:%d (up to %d clients)", WIFI_SSID, TCP_PORT, MAX_CLIENTS);
}