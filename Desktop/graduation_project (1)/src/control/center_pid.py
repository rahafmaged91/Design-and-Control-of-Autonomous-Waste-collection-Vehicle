"""
control/center_pid.py
=======================
PID controller (millimetre error, hysteretic deadband) used only by
STATE 2a (AprilTag centering). Kept separate from control/pid.py's
PIDController because it works in real-world units and its gains are
independently GUI-tunable from the "AprilTag PID" panel.
"""

import time

class CenterPIDController:
    """PID with anti-windup + a HYSTERETIC deadband, used by STATE 2a
    (AprilTag centering). Kept as its own class (distinct from the
    lane-keeping PIDController above) because it operates on millimetres,
    not a normalized [-1..1] error, and its gains are independently
    GUI-tunable from the "AprilTag PID" panel.

    ROOT-CAUSE FIX for "vehicle moves away instead of settling within the
    margin": the previous version, on every frame the error was inside the
    deadband, reset `_prev_error` to 0.0 (instead of to the small, actual
    error). The very next frame the error stepped back outside the
    deadband - even by a tiny, genuine amount, or purely from camera/tag
    detection noise - the derivative term computed
        d = kd * (error - _prev_error) / dt = kd * (error - 0.0) / dt
    which is a huge PHANTOM derivative kick (it measures a jump from 0 to
    `error`, not the true frame-to-frame change), scaled by 1/dt. That
    spike, added to a otherwise-small P term, was large enough to command a
    real, visible motor move - so the vehicle would "settle", cross into
    the deadband, get flushed, then immediately lurch again on the next
    frame that ticked barely outside it. This is a classic derivative-kick
    /  limit-cycle bug, and it's fixed two ways:

      1) `_prev_error` is now kept as the REAL last error (not force-zeroed)
         while inside the deadband, so the derivative term never sees an
         artificial jump from 0 when the error re-emerges from noise.
      2) A hysteretic (Schmitt-trigger) deadband is used: once inside the
         deadband, the error must exceed `deadband * exit_ratio` (a wider
         band) before the controller will start commanding again. This
         means small sensor/detection jitter right at the boundary can no
         longer repeatedly flip the controller in and out of the deadband
         and re-trigger corrective motion - the vehicle only leaves "settled"
         once it has genuinely drifted, not because of a single noisy frame.

    NOTE: this class's `update()` correctly returns 0.0 while inside the
    deadband - that part was already right. The "moves away instead of
    settling" symptom reported against STATE 2 turned out to live one
    level up, in MissionRunner._tick_centering(): under position-mode
    drive control (PID1/PID2 + SET1/SET2 as absolute wheel-degree
    targets), commanding "0 degrees" doesn't mean "stay put", it means
    "return to the zeroed encoder origin". See the patch note at the top
    of this file and the comments in _tick_centering() for the actual fix.
    """

    def __init__(self, kp, ki, kd, out_min, out_max, deadband=0.0, max_dt=0.5,
                 deadband_exit_ratio=1.6):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.deadband = deadband
        self.max_dt = max_dt
        # Hysteresis: must exceed deadband * this ratio to LEAVE the
        # deadband once inside it (Schmitt trigger, prevents boundary
        # chatter/limit-cycling from sensor noise).
        self.deadband_exit_ratio = max(1.0, deadband_exit_ratio)

        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None
        self._in_deadband = True   # start "settled" (no motion commanded yet)

        self.last_p = 0.0
        self.last_i = 0.0
        self.last_d = 0.0
        self.last_error = 0.0
        self.last_output = 0.0

        self._recompute_integral_limit()

    def _recompute_integral_limit(self):
        if self.ki != 0:
            self._integral_limit = max(abs(self.out_min), abs(self.out_max)) / abs(self.ki)
        else:
            self._integral_limit = float("inf")
        self._integral = max(-self._integral_limit, min(self._integral_limit, self._integral))

    def set_gains(self, kp=None, ki=None, kd=None):
        if kp is not None:
            self.kp = kp
        if ki is not None:
            self.ki = ki
        if kd is not None:
            self.kd = kd
        self._recompute_integral_limit()

    def set_deadband(self, deadband):
        self.deadband = deadband

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None
        self._in_deadband = True
        self.last_p = self.last_i = self.last_d = 0.0
        self.last_error = self.last_output = 0.0

    def update(self, error, now=None):
        now = now if now is not None else time.time()
        dt = 0.0 if self._prev_time is None else min(now - self._prev_time, self.max_dt)
        self._prev_time = now

        # Hysteretic deadband: while already settled, require a bigger
        # error (deadband * exit_ratio) before treating it as "really
        # moved" - this is what stops noise right at the boundary from
        # repeatedly kicking the controller back into motion.
        active_threshold = (self.deadband * self.deadband_exit_ratio
                            if self._in_deadband else self.deadband)

        if abs(error) < active_threshold:
            self._in_deadband = True
            self._integral = 0.0
            # Keep _prev_error as the REAL current error (not force-zeroed)
            # so that if/when we exit the deadband next frame, the
            # derivative term sees the true frame-to-frame delta instead of
            # a phantom jump from 0 - this is the fix for the "moves away
            # instead of settling" bug (see class docstring).
            self._prev_error = error
            self.last_p = self.last_i = self.last_d = 0.0
            self.last_error = error
            self.last_output = 0.0
            return 0.0

        self._in_deadband = False

        p_term = self.kp * error

        tentative_integral = self._integral + (error * dt if dt > 0 else 0.0)
        tentative_integral = max(-self._integral_limit, min(self._integral_limit, tentative_integral))
        i_term = self.ki * tentative_integral

        d_term = 0.0
        if dt > 0:
            d_term = self.kd * (error - self._prev_error) / dt
        self._prev_error = error

        raw_output = p_term + i_term + d_term
        output = max(self.out_min, min(self.out_max, raw_output))

        saturated_high = raw_output > self.out_max
        saturated_low = raw_output < self.out_min
        pushing_further = (saturated_high and error > 0) or (saturated_low and error < 0)
        if not pushing_further:
            self._integral = tentative_integral

        self.last_p, self.last_i, self.last_d = p_term, i_term, d_term
        self.last_error = error
        self.last_output = output
        return output


