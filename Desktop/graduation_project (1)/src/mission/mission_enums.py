"""
mission/mission_enums.py
==========================
Enums (and one control-flow exception) shared by the mission state
machine: overall MissionState, the gripper sub-phase, and
GripperAborted, used to unwind the gripper sequence cleanly on abort.
"""

from enum import Enum, auto

# ─── STATE 2 / STATE 3 support classes (AprilTag centering + gripper) ───────

class MissionState(Enum):
    IDLE               = auto()   # app just started, state 1 not running yet
    LANE_FOLLOWING     = auto()   # STATE 1: lane tracker driving, watching
                                   # the 8080 stream for a tag + the PC cam
                                   # for a stop sign (stub)
    TAG_TRANSITION     = auto()   # bridge: tag confirmed, STOP sent, fixed
                                   # pause before centering starts
    CENTERING          = auto()   # STATE 2a: PID-centering on the AprilTag
    STOPSIGN_HALT      = auto()   # STATE 2b: momentary stop (stop sign)
    GRIPPER            = auto()   # STATE 3: gripper sequence
    AWAITING_RESET     = auto()   # cycle done, waiting for user confirmation
    ERROR              = auto()


class GripperPhase(Enum):
    ENABLE_PID   = auto()
    HOMING       = auto()
    RESET        = auto()
    RAMP_UP      = auto()
    HOLD_MAX     = auto()
    RAMP_DOWN    = auto()
    DONE         = auto()


class GripperAborted(Exception):
    """Raised internally to unwind the gripper sequence cleanly when the
    background thread is told to stop (STOP/E-STOP pressed, or another
    command took over)."""
    pass


