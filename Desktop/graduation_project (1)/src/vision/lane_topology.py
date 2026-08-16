"""
vision/lane_topology.py
=========================
Turns the raw, per-frame boundary list from LaneDetector into a stable
notion of "lanes": sorts boundaries, pairs them into lanes of plausible
width, classifies each lane's type ("2_blacks" vs "black_yellow"),
calibrates lane width / external-space geometry on the first confirmed
frame, and keeps short-term memory of boundaries that drift out of view
so the vehicle can recover instead of stopping.
"""

import numpy as np

from config import clamp

# ─── Lane topology + memory + color-aware lane typing ───────────────────────

class LaneTopologyModel:
    """
    Turns raw, COLOR-TAGGED boundary detections into a persistent
    multi-lane world model.

    * Boundaries are tracked frame-to-frame with EMA smoothing and a
      confidence counter -> short-term MEMORY: a boundary that leaves the
      field of view stays alive for `memory_frames`, is shifted along with
      the visible boundaries (median shift), and can be steered back to.
      Every boundary also remembers its detected COLOR ("black"/"yellow").
    * The typical lane width is learned (EMA) and used to re-synthesize a
      missing partner boundary, and to reject implausible gaps. For a
      black_yellow lane, the SIGNED offset from the yellow line to the
      black edge (`yellow_black_side`) is learned too, so a re-synthesized
      partner lands on the historically-correct side of the yellow line
      rather than an arbitrary guess.
    * ROBUST "HOME LANE" MEMORY: on the very first frame the vehicle starts
      driving, the lane it is standing in is captured as `home_lane_width`
      (its own dedicated, slow-moving EMA - independent of the generic,
      road-wide `lane_width` estimate) plus a persistent `desired_anchor_x`.
      Every frame the chosen lane is fully tracked, `home_lane_width` is
      refreshed a little more (slow EMA, so a momentary bad detection can't
      swing it) - so the model always has a dedicated, trustworthy width
      estimate for the SPECIFIC lane being followed, not just "some lane
      gap seen somewhere in the image". This dedicated width is what is
      used (in preference to the generic `lane_width`) to re-synthesize a
      missing border and to compute the recovery target when both borders
      vanish, which keeps the vehicle anchored to the lane it actually
      started in instead of drifting into a neighbouring lane of a
      different width.
    * Lanes are named left_edge / center / right_edge (generalizes to any
      lane count) and the vehicle's current lane / edge position is
      classified every frame. Each lane also gets a LANE TYPE:
        - "black_yellow": one border yellow, the other black.
        - "2_blacks"    : both borders black.
    * PRIORITY-AWARE STATE: for the chosen lane,
        - both borders real & seen this frame        -> TRACKING
        - black_yellow lane, yellow seen, black not   -> PREDICTING
          (target keeps coming from the yellow line + the learned
          offset; not urgent, no strong steer bias)
        - the CRITICAL border missing (yellow in a black_yellow lane,
          either edge in a 2_blacks lane)             -> RECOVERING
          (`missing_side` says which way to steer to bring it back)
        - both borders gone                           -> RECOVERING
          toward the remembered lane anchor (`recovery_x`), or LOST if
          there's no memory left at all.
    * FUTURE LANE SWITCHING is architected in: `request_lane_change(dir)`
      moves a persistent desired-lane anchor (an x position, so it survives
      lanes appearing/disappearing); while desired != current the state is
      CHANGING_LANE and the PID target becomes the desired lane's center.
      A future state machine only needs to call request_lane_change().
    """

    STATE_INIT       = "INIT"
    STATE_TRACKING   = "TRACKING"
    STATE_PREDICTING = "PREDICTING"   # currently unused: since the vehicle
                                       # follows the yellow line directly,
                                       # only the yellow line's visibility
                                       # gates control - kept as a slot for
                                       # a future degraded mode.
    STATE_CHANGING   = "CHANGING_LANE"
    STATE_RECOVERING = "RECOVERING"
    STATE_LOST       = "LANE_LOST"

    def __init__(self):
        self.boundaries = []        # [{"x","conf","virtual","seen","color"}]
        self.lane_width = None      # learned typical lane width (px), road-wide
        # dedicated, slow-EMA width of the SPECIFIC lane being followed -
        # this is the "home lane" memory the vehicle anchors to.
        self.home_lane_width = None
        self.yellow_black_side = None   # +1: black is right of yellow, -1: left
        self.locked_type = None     # SYSTEM TYPE locked by the calibration image
        self.state = self.STATE_INIT
        self.desired_anchor_x = None   # persistent x of the desired lane center
        self.target_offset = 0.0       # -1..+1 inside the lane (GUI slider)
        # tuning
        self.ema = 0.55
        self.home_width_ema = 0.10     # slow EMA for the dedicated home-lane width
        self.max_conf = 10
        self.memory_frames = 25        # ~ how long an unseen boundary survives
        self.match_tol_frac = 0.08

    def reset(self):
        self.boundaries = []
        self.lane_width = None
        self.home_lane_width = None
        self.yellow_black_side = None
        self.locked_type = None
        self.state = self.STATE_INIT
        self.desired_anchor_x = None

    def lock_system_type(self, lane_type):
        """Lock the SYSTEM TYPE from the calibration image. After locking,
        the behavioral branch (yellow-priority vs. plain two-border) always
        follows the calibrated type, so a single-frame color misdetection
        (yellow flickering below the saturation floor, or a stray yellowish
        object) can't flip the control strategy mid-drive."""
        if lane_type in ("black_yellow", "2_blacks"):
            self.locked_type = lane_type

    def seed_home_lane(self, width_px, anchor_x):
        """Called once, at calibration time (the first confirmed, fully
        visible frame): seeds the dedicated home-lane width memory so
        re-synthesis/recovery has a trustworthy value to use from frame 1,
        instead of waiting for the generic road-wide EMA to catch up."""
        if width_px and width_px > 1:
            self.home_lane_width = float(width_px)
        if anchor_x is not None:
            self.desired_anchor_x = float(anchor_x)

    # -- future feature hook ---------------------------------------------------

    def request_lane_change(self, direction):
        """direction: -1 = one lane to the left, +1 = one lane to the right.
        (Experimental hook for the future lane-switch / state-machine layer.)"""
        w = self.home_lane_width or self.lane_width
        if self.desired_anchor_x is None or not w:
            return False
        self.desired_anchor_x += direction * w
        return True

    def shift_desired_lane(self, direction, frame_w):
        """Used by the startup confirmation dialog ('or switch')."""
        w = self.home_lane_width or self.lane_width
        if w:
            base = self.desired_anchor_x if self.desired_anchor_x is not None \
                   else frame_w / 2.0
            self.desired_anchor_x = base + direction * w
            return True
        return False

    @staticmethod
    def _lane_type(color_a, color_b):
        if "yellow" in (color_a, color_b) and color_a != color_b:
            return "black_yellow"
        if color_a == "black" and color_b == "black":
            return "2_blacks"
        if color_a == "yellow" and color_b == "yellow":
            return "2_yellows"
        return "unknown"

    # -- main update -------------------------------------------------------------

    def update(self, xs_colored, frame_w):
        """xs_colored: list of (x, color) tuples from the detector."""
        cx = frame_w / 2.0
        tol = max(22.0, self.match_tol_frac * frame_w)
        xs = [x for x, _ in xs_colored]

        # 1) match detections to remembered boundaries (greedy nearest)
        cand = []
        for bi, b in enumerate(self.boundaries):
            for xi, x in enumerate(xs):
                d = abs(x - b["x"])
                if d <= tol:
                    cand.append((d, bi, xi))
        cand.sort(key=lambda c: c[0])
        used_b, used_x, shifts = set(), set(), []
        for d, bi, xi in cand:
            if bi in used_b or xi in used_x:
                continue
            used_b.add(bi)
            used_x.add(xi)
            b = self.boundaries[bi]
            shifts.append(xs[xi] - b["x"])
            b["x"] = self.ema * xs[xi] + (1 - self.ema) * b["x"]
            b["conf"] = min(b["conf"] + 2, self.max_conf)
            b["virtual"] = False
            b["seen"] = True
            b["color"] = xs_colored[xi][1]     # trust the fresh detection
        median_shift = float(np.median(shifts)) if shifts else 0.0

        # 2) memory: unseen boundaries drift with the pack, confidence decays
        for bi, b in enumerate(self.boundaries):
            if bi not in used_b:
                b["x"] += median_shift
                b["conf"] -= 1
                b["seen"] = False
        self.boundaries = [b for b in self.boundaries
                           if b["conf"] > 0 and -0.7 * frame_w < b["x"] < 1.7 * frame_w]

        # 3) brand new boundaries
        for xi, (x, color) in enumerate(xs_colored):
            if xi not in used_x:
                self.boundaries.append({"x": float(x), "conf": 3,
                                        "virtual": False, "seen": True,
                                        "color": color})

        # 4) merge near-duplicates, sort
        self.boundaries.sort(key=lambda b: b["x"])
        merged = []
        for b in self.boundaries:
            if merged and abs(b["x"] - merged[-1]["x"]) < 0.35 * tol:
                keep = max(merged[-1], b, key=lambda q: q["conf"])
                keep["x"] = (merged[-1]["x"] + b["x"]) / 2.0
                keep["seen"] = merged[-1].get("seen", False) or b.get("seen", False)
                merged[-1] = keep
            else:
                merged.append(b)
        self.boundaries = merged

        usable = [b for b in self.boundaries if b["conf"] >= 2]

        # 5) learn lane width from plausible gaps. (The signed yellow->black
        #    offset is deliberately NOT learned here: on a 3-line road the
        #    two lanes have OPPOSITE yellow->black relations and a global
        #    scan would let the last pair clobber the chosen lane's one. It
        #    is learned in step 11, from the DESIRED lane only.)
        gaps = []
        for a, b in zip(usable, usable[1:]):
            g = b["x"] - a["x"]
            lo = 0.12 * frame_w if not self.lane_width else 0.55 * self.lane_width
            hi = 0.85 * frame_w if not self.lane_width else 1.7 * self.lane_width
            if lo < g < hi:
                gaps.append(g)
        if gaps:
            g = float(np.median(gaps))
            self.lane_width = g if self.lane_width is None \
                else 0.9 * self.lane_width + 0.1 * g

        # 5b) the width actually used for synthesis/recovery: the dedicated,
        #     slow-moving HOME LANE width when we have one (learned only
        #     from the specific lane being followed, step 11), falling back
        #     to the generic road-wide estimate before that memory exists.
        effective_width = self.home_lane_width or self.lane_width

        # 6) re-synthesize a missing partner boundary (memory of the road).
        #    POSITION: rebuild the lane the vehicle was actually tracking.
        #    The persistent desired-lane anchor says on which side of the
        #    lone survivor the lane was - without it, a divider line (e.g.
        #    the yellow) could get its partner synthesized on the WRONG
        #    side, silently re-anchoring the vehicle to the neighbour lane.
        if len(usable) == 1 and effective_width:
            b = usable[0]
            if (self.desired_anchor_x is not None
                    and abs(self.desired_anchor_x - b["x"]) >
                    0.15 * effective_width):
                side = 1 if self.desired_anchor_x > b["x"] else -1
            elif b.get("color") == "yellow" and self.yellow_black_side is not None:
                side = self.yellow_black_side
            elif b.get("color") == "black" and self.yellow_black_side is not None:
                side = -self.yellow_black_side
            else:
                side = 1 if b["x"] < cx else -1   # generic: lane around camera
            # COLOR: the partner of a yellow divider is a black edge; the
            # partner of a black edge is yellow only in a yellow system.
            if b.get("color") == "yellow":
                new_color = "black"
            else:
                new_color = ("yellow" if (self.locked_type == "black_yellow"
                                          or self.yellow_black_side is not None)
                             else "black")
            self.boundaries.append({"x": b["x"] + side * effective_width,
                                    "conf": 2, "virtual": True, "seen": False,
                                    "color": new_color})
            self.boundaries.sort(key=lambda q: q["x"])
            usable = [q for q in self.boundaries if q["conf"] >= 2]

        # 7) lanes = plausible adjacent gaps
        lanes = []
        gate_width = effective_width or self.lane_width
        for a, b in zip(usable, usable[1:]):
            g = b["x"] - a["x"]
            if gate_width and not (0.5 * gate_width < g < 1.8 * gate_width):
                continue
            if not gate_width and not (0.1 * frame_w < g < 0.9 * frame_w):
                continue
            lanes.append((a["x"], b["x"]))

        result = {
            "lanes": lanes,
            "lane_names": self._names(len(lanes)),
            "boundaries": [(b["x"], b["conf"], b.get("virtual", False),
                           b.get("color", "black")) for b in self.boundaries],
            "lane_width": self.lane_width,
            "home_lane_width": self.home_lane_width,
            "current_lane": None,
            "position": "unknown",
            "desired_lane": None,
            "target_x": None,
            "lane_type": "unknown",
            "system_type": self.locked_type,
            "yellow_x": None,
            "black_x": None,
            "missing_side": 0,        # -1 line faded on the LEFT, +1 RIGHT
            "recovery_x": None,       # remembered x to steer back to
            "state": self.STATE_LOST,
        }

        if not lanes:
            # No usable lane this frame. Before declaring LANE_LOST, expose
            # the short-term memory so the controller can steer back to
            # where the road was instead of stopping immediately. Prefer
            # the persistent desired-lane anchor (the specific lane the
            # vehicle started in) over a generic median of whatever
            # boundaries happen to still be in memory - that median can be
            # dragged toward a neighbouring lane if this lane's own borders
            # faded first.
            rec_x = None
            if self.desired_anchor_x is not None:
                rec_x = float(self.desired_anchor_x)
            elif self.boundaries:
                rec_x = float(np.median([b["x"] for b in self.boundaries]))
            if rec_x is not None:
                self.state = self.STATE_RECOVERING
                result["recovery_x"] = rec_x
                result["missing_side"] = -1 if rec_x < cx else 1
            else:
                self.state = self.STATE_LOST
            result["state"] = self.state
            return result

        # 8) classify the vehicle's current lane / edge position
        current = None
        for i, (l, r) in enumerate(lanes):
            if l <= cx <= r:
                current = i
                break
        if current is None:
            result["position"] = ("left_edge (left of leftmost line)"
                                  if cx < lanes[0][0]
                                  else "right_edge (right of rightmost line)")
            current = 0 if cx < lanes[0][0] else len(lanes) - 1
        else:
            result["position"] = result["lane_names"][current]
        result["current_lane"] = current

        # 9) desired lane follows a persistent x anchor
        if self.desired_anchor_x is None:
            self.desired_anchor_x = sum(lanes[current]) / 2.0
        centers = [(l + r) / 2.0 for l, r in lanes]
        desired = int(np.argmin([abs(c - self.desired_anchor_x)
                                 for c in centers]))
        self.desired_anchor_x = centers[desired]     # re-anchor (drifts w/ road)
        result["desired_lane"] = desired

        # 10) find the two boundary records bordering the desired lane, and
        #     classify the lane's TYPE from their colors. The BEHAVIORAL
        #     type is the calibration-locked one when set (spec: the system
        #     type is set by the first calibration image); the live type is
        #     still reported for the GUI/debug overlay.
        l, r = lanes[desired]
        bl = min(self.boundaries, key=lambda b: abs(b["x"] - l)) \
            if self.boundaries else None
        br = min(self.boundaries, key=lambda b: abs(b["x"] - r)) \
            if self.boundaries else None
        lane_type = self._lane_type(bl.get("color") if bl else None,
                                    br.get("color") if br else None)
        result["lane_type"] = lane_type
        result["system_type"] = self.locked_type
        eff_type = self.locked_type if self.locked_type else lane_type

        yb = bl if bl.get("color") == "yellow" else (
            br if br.get("color") == "yellow" else None)
        ob = bl if bl.get("color") == "black" else (
            br if br.get("color") == "black" else None)
        result["yellow_x"] = yb["x"] if yb else None
        result["black_x"] = ob["x"] if ob else None

        half = (r - l) / 2.0
        offset_px = clamp(self.target_offset, -1.0, 1.0) * 0.85 * half

        if eff_type == "black_yellow" and yb is not None:
            # FOLLOW THE YELLOW LINE: drive so the yellow line itself sits
            # at the target position, instead of the midpoint of the lane
            # it divides. The offset slider still allows a deliberate
            # lateral bias away from the line (e.g. to hug one side of it).
            result["target_x"] = yb["x"] + offset_px
        else:
            result["target_x"] = (l + r) / 2.0 + offset_px

        # 10b) HOME LANE WIDTH: whenever the chosen lane's two boundaries
        #      are BOTH real & currently seen (the only time the measured
        #      gap is trustworthy), refresh the dedicated, slow-moving
        #      home-lane width memory. This is deliberately a *slower* EMA
        #      than the generic `lane_width` learner above, and is scoped
        #      to the DESIRED lane only, so it stays a faithful memory of
        #      "how wide is MY lane" even on a road where other lanes have
        #      a different width - this is what re-synthesis/recovery
        #      (steps 6/11) lean on to stay anchored to the home lane
        #      instead of drifting toward a neighbour.
        bl_real = bl is not None and bl.get("seen") and not bl.get("virtual")
        br_real = br is not None and br.get("seen") and not br.get("virtual")
        if bl_real and br_real:
            gap_now = r - l
            if self.home_lane_width is None:
                self.home_lane_width = gap_now
            else:
                self.home_lane_width = ((1 - self.home_width_ema) * self.home_lane_width
                                        + self.home_width_ema * gap_now)

        # 11) priority-aware missing/critical detection for the chosen lane.
        #     Following the yellow line directly means the black edge no
        #     longer gates control at all: only whether the YELLOW LINE
        #     itself is real & currently seen matters.
        missing = 0
        if eff_type == "black_yellow" and yb is not None:
            yellow_real = (not yb.get("virtual", False)) and yb.get("seen", False)
            yellow_is_left = (yb is bl)
            if not yellow_real:
                # CRITICAL: the line being FOLLOWED is gone from view.
                missing = -1 if yellow_is_left else 1
                self.state = self.STATE_RECOVERING
            else:
                # Learn/refresh this lane's signed yellow->black relation
                # (used by step 6 re-synthesis and the full-loss branch
                # below) whenever a real black edge is also seen alongside
                # the yellow line. Learning only from the desired lane
                # keeps it correct on 3-line roads with opposite relations
                # either side of the divider.
                if ob and ob.get("seen") and not ob.get("virtual"):
                    self.yellow_black_side = 1 if ob["x"] > yb["x"] else -1
                self.state = self.STATE_TRACKING
        elif eff_type == "black_yellow":
            # Calibrated as a yellow system, but no yellow border exists on
            # this lane even in memory: the followed line is FULLY lost
            # (worse than merely unseen). Recover toward where it should
            # be, using the learned lane relation when available.
            if self.yellow_black_side is not None:
                missing = -self.yellow_black_side   # yellow sits opposite black
            else:
                missing = -1 if bl["conf"] <= br["conf"] else 1
            self.state = self.STATE_RECOVERING
        else:
            l_real = (not bl.get("virtual", False)) and bl.get("seen", False)
            r_real = (not br.get("virtual", False)) and br.get("seen", False)
            if not l_real and not r_real:
                missing = -1 if bl["conf"] <= br["conf"] else 1
            elif not l_real:
                missing = -1
            elif not r_real:
                missing = 1
            self.state = self.STATE_RECOVERING if missing != 0 \
                else self.STATE_TRACKING
        result["missing_side"] = missing
        result["home_lane_width"] = self.home_lane_width

        if desired != current:
            self.state = self.STATE_CHANGING
        result["state"] = self.state
        return result

    @staticmethod
    def _names(n):
        if n <= 0:
            return []
        if n == 1:
            return ["single_lane"]
        if n == 2:
            return ["left_edge", "right_edge"]
        if n == 3:
            return ["left_edge", "center", "right_edge"]
        mids = [f"center_{i}" for i in range(1, n - 1)]
        return ["left_edge"] + mids + ["right_edge"]

