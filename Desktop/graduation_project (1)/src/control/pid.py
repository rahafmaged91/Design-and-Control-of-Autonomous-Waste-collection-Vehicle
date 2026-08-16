"""
control/pid.py
===============
Generic PID controller with anti wind-up (clamped + conditional
integration) and an error-margin deadband. Used both for lane-keeping
steering and, separately, for AprilTag centering (see control/center_pid.py).
"""

from config import clamp

# ─── PID controller (anti wind-up + error margin) ───────────────────────────

class PIDController:
    """
    PID with:
      * deadband (error margin): inside it the output is exactly 0 and the
        integrator is flushed -> vehicle is commanded perfectly straight.
      * anti wind-up: the integral term is clamped AND conditionally
        integrated (integration is skipped when the output is saturated and
        the error would push it further into saturation).
      * derivative on error with dt scaling.
    Output is normalized to [-1 .. +1].
    """

    def __init__(self, kp=1.6, ki=0.05, kd=0.35,
                 deadband=0.04, integral_limit=0.5, output_limit=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.deadband = deadband
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.reset()

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_valid = False

    def update(self, error, dt):
        dt = max(dt, 1e-3)

        # Error margin: close enough to the target -> go straight, and
        # flush the integrator so it can't wind up while we sit centered.
        if abs(error) <= self.deadband:
            self._integral = 0.0
            self._prev_error = error
            self._prev_valid = True
            return 0.0

        p = self.kp * error

        d = 0.0
        if self._prev_valid:
            d = self.kd * (error - self._prev_error) / dt
        self._prev_error = error
        self._prev_valid = True

        # Provisional output without new integration, to decide whether
        # integrating would just wind up against saturation.
        provisional = p + self.ki * self._integral + d
        saturated_hi = provisional >= self.output_limit and error > 0
        saturated_lo = provisional <= -self.output_limit and error < 0
        if not (saturated_hi or saturated_lo):
            self._integral += error * dt
            self._integral = clamp(self._integral,
                                   -self.integral_limit, self.integral_limit)

        out = p + self.ki * self._integral + d
        return clamp(out, -self.output_limit, self.output_limit)

