"""
vision/lane_detector.py
========================
Per-frame vision pipeline: grayscale + HSV segmentation of the black
road-edge lines and the yellow lane-divider line, temporal low-pass
filtering, scan-line sampling and polynomial-fit boundary tracking.
See the project README ("How It Works") for the full pipeline
description.

Requires opencv-python and numpy.
"""

import cv2
import numpy as np

from config import clamp

# ─── Lane detection (black edges + yellow divider) ──────────────────────────

class LaneDetector:
    """
    Robust detector for THICK black boundary lines AND a thinner painted
    YELLOW divider line on a bright floor. Runs two parallel segmentation
    pipelines (grayscale dark-mask for black, HSV band-pass for yellow)
    through the SAME downstream scan-line / track-association / polynomial
    fit machinery, and tags every resulting boundary with its color.

    Pipeline: grayscale -> Gaussian blur -> temporal low-pass filter
    (adjustable alpha, EMA between consecutive frames) -> ROI band (top
    height adjustable from the GUI) -> {dark mask, yellow HSV mask} ->
    morphology -> connected component filtering (area + elongation gates)
    -> scan-line run extraction (per-color width gate) over `num_rows` rows
    (the RESOLUTION slider: more rows = more small segments per boundary =
    smoother interpolation) -> bottom-up track association into small
    semi-parallel segments -> polynomial fit per track (interpolation /
    extrapolation through curves and occlusions).
    """

    def __init__(self):
        # All of these are refreshed live from the GUI every frame.
        self.black_thresh   = 70     # 0..255, pixel darker than this = "black"
        self.roi_top_ratio  = 0.55   # the adjustable horizontal line height
        self.roi_bottom_ratio = 0.98
        self.num_rows       = 14     # scan rows inside the ROI (RESOLUTION)
        self.blur_ksize     = 5
        self.min_line_w     = 5      # accepted black-run width in px (thick lines!)
        self.max_line_w     = 110
        self.min_comp_area  = 120    # connected-component area gate (black)
        self.min_track_rows = 4      # a boundary must appear in >= N scan rows
        self.show_edges     = True   # overlay Canny edges on the debug view
        # Temporal low-pass filter between frames (EMA on the blurred
        # grayscale image). alpha = weight of the CURRENT frame:
        #   1.0 -> filter off, 0.05 -> very heavy smoothing.
        self.lpf_alpha      = 0.7
        self._lpf_prev      = None   # float32 accumulator

        # Yellow divider line: HSV band-pass. Hue/value bounds are fixed
        # (yellow paint is a narrow, predictable hue band); saturation
        # floor is live-adjustable from the GUI since it's what varies
        # most with lighting.
        self.yellow_hue_lo   = 15
        self.yellow_hue_hi   = 40
        self.yellow_sat_min  = 80
        self.yellow_val_min  = 90
        self.min_line_w_y    = 3      # yellow line is thinner than black
        self.max_line_w_y    = 70
        self.min_comp_area_y = 60

    # -- low level helpers ---------------------------------------------------

    @staticmethod
    def _runs(binary_row):
        """Return [(start, end_exclusive), ...] runs of nonzero pixels."""
        idx = np.flatnonzero(binary_row)
        if idx.size == 0:
            return []
        splits = np.where(np.diff(idx) > 1)[0]
        starts = np.concatenate(([idx[0]], idx[splits + 1]))
        ends = np.concatenate((idx[splits], [idx[-1]]))
        return list(zip(starts.tolist(), (ends + 1).tolist()))

    def reset_temporal_filter(self):
        self._lpf_prev = None

    def _temporal_lpf(self, blurred):
        """Exponential moving average between consecutive frames."""
        alpha = clamp(float(self.lpf_alpha), 0.05, 1.0)
        if alpha >= 0.999:
            self._lpf_prev = None          # filter disabled
            return blurred
        f32 = blurred.astype(np.float32)
        if self._lpf_prev is None or self._lpf_prev.shape != f32.shape:
            self._lpf_prev = f32
        else:
            self._lpf_prev = alpha * f32 + (1.0 - alpha) * self._lpf_prev
        return self._lpf_prev.astype(np.uint8)

    def _clean_mask(self, mask, band_h, w, min_area):
        """Shared morphology + elongation/area connected-component filter
        used by both the black mask and the yellow mask."""
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (5, 7)))
        n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        keep = np.zeros(n_lbl, dtype=bool)
        min_span = max(6, int(0.22 * band_h))
        for i in range(1, n_lbl):
            area = stats[i, cv2.CC_STAT_AREA]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            if area < min_area:
                continue
            if bh < min_span:                 # not elongated -> clutter
                continue
            if bw > 0.75 * w and bh < 0.5 * band_h:
                continue                      # wide flat smear (shadow band)
            keep[i] = True
        return np.where(keep[labels], 255, 0).astype(np.uint8)

    def _scan_rows(self, clean, top, row_ys, band_h, min_w_gate, max_w_gate):
        """Scan-line sampling of a cleaned binary mask -> per-row candidate
        points, with a perspective-aware width gate (lines get thinner
        further away)."""
        row_points = []
        for y in row_ys:
            r = clean[y - top]
            pts = []
            for s, e in self._runs(r):
                run_w = e - s
                depth = (y - top) / max(band_h, 1)      # 0 far .. 1 near
                lo = max(2, min_w_gate * (0.4 + 0.6 * depth))
                hi = max_w_gate * (0.5 + 0.7 * depth)
                if lo <= run_w <= hi:
                    pts.append((s + e) / 2.0)
            row_points.append((int(y), pts))
        return row_points

    def _extract_boundaries(self, row_points, w, control_y, min_rows_needed):
        """Associate scan-row points into tracks and fit a polynomial to
        each -> list of boundary dicts with x_ctrl + coeffs (color-agnostic;
        caller tags the color)."""
        tracks = self._associate(row_points, w)
        boundaries = []
        for t in tracks:
            n = len(t["xs"])
            if n < min_rows_needed:
                continue
            deg = 2 if n >= 6 else 1
            try:
                t["coeffs"] = np.polyfit(t["ys"], t["xs"], deg)
            except Exception:
                continue
            x_ctrl = float(np.polyval(t["coeffs"], control_y))
            if -0.5 * w < x_ctrl < 1.5 * w:
                t["x_ctrl"] = x_ctrl
                boundaries.append(t)
        return boundaries

    # -- main ------------------------------------------------------------------

    def process(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        k = self.blur_ksize | 1                      # force odd
        blurred = cv2.GaussianBlur(gray, (k, k), 0)

        # temporal low-pass filter between frames (adjustable alpha)
        blurred = self._temporal_lpf(blurred)

        top = int(h * clamp(self.roi_top_ratio, 0.05, 0.9))
        bottom = int(h * clamp(self.roi_bottom_ratio, 0.2, 1.0))
        bottom = max(bottom, top + 10)
        band_h = bottom - top
        control_y = bottom - 1

        n_rows = int(clamp(self.num_rows, 4, 48))
        row_ys = np.linspace(bottom - 1, top + 1, n_rows).astype(int)
        min_rows_needed = max(3, min(self.min_track_rows, int(0.25 * n_rows)))

        # ---- BLACK pipeline ------------------------------------------------
        band = blurred[top:bottom]
        _, dark = cv2.threshold(band, self.black_thresh, 255,
                                cv2.THRESH_BINARY_INV)
        clean_black = self._clean_mask(dark, band_h, w, self.min_comp_area)
        min_w_gate = max(2, int(self.min_line_w))
        max_w_gate = max(min_w_gate + 1, int(self.max_line_w))
        rows_black = self._scan_rows(clean_black, top, row_ys, band_h,
                                     min_w_gate, max_w_gate)
        black_boundaries = self._extract_boundaries(rows_black, w, control_y,
                                                     min_rows_needed)
        for t in black_boundaries:
            t["color"] = "black"

        # ---- YELLOW pipeline -------------------------------------------------
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        band_hsv = hsv[top:bottom]
        lower = np.array([self.yellow_hue_lo, self.yellow_sat_min,
                          self.yellow_val_min], dtype=np.uint8)
        upper = np.array([self.yellow_hue_hi, 255, 255], dtype=np.uint8)
        yellow_raw = cv2.inRange(band_hsv, lower, upper)
        clean_yellow = self._clean_mask(yellow_raw, band_h, w,
                                        self.min_comp_area_y)
        min_w_gate_y = max(2, int(self.min_line_w_y))
        max_w_gate_y = max(min_w_gate_y + 1, int(self.max_line_w_y))
        rows_yellow = self._scan_rows(clean_yellow, top, row_ys, band_h,
                                      min_w_gate_y, max_w_gate_y)
        yellow_boundaries = self._extract_boundaries(rows_yellow, w, control_y,
                                                      min_rows_needed)
        for t in yellow_boundaries:
            t["color"] = "yellow"

        # ---- combine ---------------------------------------------------------
        boundaries = black_boundaries + yellow_boundaries
        boundaries.sort(key=lambda t: t["x_ctrl"])

        debug = self._draw_debug(frame, top, bottom, row_ys,
                                 rows_black, rows_yellow, boundaries,
                                 control_y, blurred)

        return {
            "found": len(boundaries) > 0,
            "boundaries": boundaries,          # tracks with x_ctrl + coeffs + color
            "boundary_xs": [(t["x_ctrl"], t["color"]) for t in boundaries],
            "control_y": control_y,
            "roi_top": top,
            "roi_bottom": bottom,
            "frame_w": w,
            "frame_h": h,
            "debug_frame": debug,
        }

    def _associate(self, row_points, frame_w):
        """Chain per-row points bottom-up into tracks using predicted-x
        nearest-neighbour matching. Tolerance follows the track's own slope
        so curved lines stay connected while separate lines never merge."""
        tol = max(16.0, 0.06 * frame_w)
        tracks = []

        for y, pts in row_points:              # already bottom -> top
            if not pts:
                for t in tracks:
                    t["miss"] += 1
                continue

            preds = []
            for t in tracks:
                if len(t["xs"]) >= 2:
                    # linear extrapolation from the last two accepted points
                    dy = t["ys"][-1] - t["ys"][-2]
                    dx = t["xs"][-1] - t["xs"][-2]
                    slope = dx / dy if dy != 0 else 0.0
                    pred = t["xs"][-1] + slope * (y - t["ys"][-1])
                else:
                    pred = t["xs"][-1]
                preds.append(pred)

            # greedy global matching
            cand = []
            for pi, x in enumerate(pts):
                for ti, pred in enumerate(preds):
                    d = abs(x - pred)
                    if d <= tol * (1 + 0.5 * tracks[ti]["miss"]):
                        cand.append((d, ti, pi))
            cand.sort(key=lambda c: c[0])
            used_t, used_p = set(), set()
            for d, ti, pi in cand:
                if ti in used_t or pi in used_p:
                    continue
                used_t.add(ti)
                used_p.add(pi)
                tracks[ti]["xs"].append(pts[pi])
                tracks[ti]["ys"].append(y)
                tracks[ti]["miss"] = 0

            for ti, t in enumerate(tracks):
                if ti not in used_t:
                    t["miss"] += 1
            for pi, x in enumerate(pts):
                if pi not in used_p:
                    tracks.append({"xs": [x], "ys": [y], "miss": 0})

        return tracks

    def _draw_debug(self, frame, top, bottom, row_ys, rows_black,
                    rows_yellow, boundaries, control_y, blurred):
        debug = frame.copy()
        h, w = debug.shape[:2]

        # optional Canny overlay for tuning (visual only)
        if self.show_edges:
            edges = cv2.Canny(blurred[top:bottom], 60, 150)
            debug[top:bottom][edges > 0] = (60, 60, 255)

        # ROI band + the adjustable horizontal scan-height line
        cv2.rectangle(debug, (0, top), (w - 1, bottom - 1), (60, 60, 60), 1)
        cv2.line(debug, (0, top), (w, top), (0, 210, 255), 2)

        for y, pts in rows_black:
            for x in pts:
                cv2.circle(debug, (int(x), int(y)), 3, (0, 165, 255), -1)
        for y, pts in rows_yellow:
            for x in pts:
                cv2.circle(debug, (int(x), int(y)), 3, (0, 255, 255), -1)

        # fitted boundary curves (interpolated), segment by segment
        ys = np.linspace(top, bottom - 1, 24)
        for t in boundaries:
            xs = np.polyval(t["coeffs"], ys)
            pts = np.array([[int(x), int(y)] for x, y in zip(xs, ys)
                            if -50 < x < w + 50], dtype=np.int32)
            color = (0, 255, 255) if t["color"] == "yellow" else (255, 80, 0)
            if len(pts) >= 2:
                cv2.polylines(debug, [pts], False, color, 2)
            cv2.circle(debug, (int(t["x_ctrl"]), control_y), 6, color, -1)

        cv2.line(debug, (w // 2, top), (w // 2, bottom), (180, 180, 180), 1)
        return debug

    # -- calibration helper ----------------------------------------------------

    def suggest_threshold(self, frame):
        """Otsu on the current ROI band -> suggested black threshold."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        top = int(h * clamp(self.roi_top_ratio, 0.05, 0.9))
        bottom = max(int(h * self.roi_bottom_ratio), top + 10)
        roi = gray[top:bottom, :]
        otsu, _ = cv2.threshold(roi, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        stats = {"min": int(roi.min()), "max": int(roi.max()),
                 "mean": float(roi.mean()), "otsu": float(otsu),
                 "roi_top": top, "roi_bottom": bottom}
        return otsu, stats

