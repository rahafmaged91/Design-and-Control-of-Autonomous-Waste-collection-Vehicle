"""
gui/app.py
===========
The Tkinter GUI: dark-themed main window wiring together manual/
autonomous driving (VehicleController), the mission state machine
(MissionRunner), the live camera preview, lane-keeping/PID/road-type
tuning panels, and the STATE 2/3 (AprilTag + gripper) sidebar.
"""

import os
import re
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from pupil_apriltags import Detector as AprilTagDetector
    APRILTAG_AVAILABLE = True
except ImportError:
    APRILTAG_AVAILABLE = False

from config import (
    clamp, STEER_CENTER, SPEED_MAX_VALUE, TCP_DEFAULT_PORT,
    ESP32_CAM_IP, ESP32_CAM_PORT, FRAMESIZE_PRESETS,
    TAG_PID_KP, TAG_PID_KI, TAG_PID_KD, TAG_ERROR_MARGIN_MM,
    GRIPPER_RAMP_STEP, GRIPPER_RAMP_INTERVAL, GRIPPER_HOLD_AT_MAX_SEC,
    GRIPPER_HOMING_STABLE_COUNT,
)
from network.tcp_client import TCPClient
from camera_utils import scan_cameras, open_camera
from vision.lane_detector import LaneDetector
from vision.camera_resolution import set_camera_resolution
from control.vehicle_controller import VehicleController
from control.lane_keep_controller import LaneKeepController
from mission.mission_runner import MissionRunner
from mission.mission_enums import MissionState, GripperPhase

# ─── GUI ────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    """
    Professional layout notes:
      * A single ttk 'clam' theme is applied app-wide (Notebook, Scrollbar,
        Combobox all themed to match the dark palette) instead of relying
        purely on raw tk widget colors, so tabs/scrollbars/dropdowns look
        consistent rather than like default-OS widgets dropped on a dark
        background.
      * STATE 1's controls are organized into a ttk.Notebook with three
        tabs - DRIVE & PATH, DETECTION & PID, ROAD TYPE - instead of three
        permanently-wide scrolling columns. Everything is still reachable
        in one or two clicks, but the window no longer needs a horizontal
        scrollbar to reach the road-type/PID panel on smaller screens.
      * The camera preview, lane status, keyboard map and command log stay
        OUTSIDE the notebook (always visible) since they're the things you
        want in view no matter which settings tab you're tuning.
      * The STATE 2/3 sidebar keeps its own always-visible layout (stream
        first, controls below) as before, just re-skinned with the shared
        theme.
    """
    DARK   = "#0d0f12"
    PANEL  = "#161920"
    CARD   = "#1e2230"
    ACCENT = "#00d4ff"
    GREEN  = "#00e676"
    RED    = "#ff4b4b"
    YELLOW = "#ffd54f"
    TEXT   = "#e0e8f0"
    MUTED  = "#5a6a7a"
    FONT   = ("Consolas", 10)

    def __init__(self):
        super().__init__()
        self.title("Vehicle RC Controller + Multi-Lane Tracking")
        self.configure(bg=self.DARK)
        self.resizable(True, True)
        self.geometry("1220x780")
        self.minsize(820, 520)

        self._setup_style()

        self.tcp = TCPClient()
        self.ctrl = VehicleController(self.tcp, status_cb=self._on_status,
                                      log_cb=self._log)
        self.mission = MissionRunner(self.tcp, self.ctrl, log_cb=self._log,
                                     restart_lane_follow_cb=self._restart_lane_follow)

        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._camera_photo = None
        self._mission_photo = None
        self._lk_params = {}
        self._params_lock = threading.Lock()
        self._last_auto_camera_index = 0
        self._last_auto_road_type = None

        self._build_ui()
        self._bind_keys()
        self.ctrl.start()
        self.mission.start()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        if not CV2_AVAILABLE:
            self._log("opencv-python/numpy not installed - lane tracking "
                      "disabled (pip install opencv-python numpy).")
        elif not PIL_AVAILABLE:
            self._log("pillow not installed - lane tracking will run without "
                      "a live camera preview (pip install pillow).")
        if not APRILTAG_AVAILABLE:
            self._log("pupil_apriltags not installed - AprilTag centering "
                      "(STATE 2) disabled (pip install pupil_apriltags).")

        self._snapshot_params()
        self._refresh_camera_preview()
        self._refresh_mission_panel()

    # ── shared ttk theme (professional look for Notebook/Scrollbar/Combobox) ─
    def _setup_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TNotebook", background=self.DARK, borderwidth=0,
                        tabmargins=(4, 4, 4, 0))
        style.configure("TNotebook.Tab", background=self.PANEL,
                        foreground=self.MUTED, padding=(14, 7),
                        font=("Consolas", 9, "bold"), borderwidth=0)
        style.map("TNotebook.Tab",
                 background=[("selected", self.CARD)],
                 foreground=[("selected", self.ACCENT)])
        style.configure("TFrame", background=self.DARK)
        style.configure("Vertical.TScrollbar", background=self.CARD,
                        troughcolor=self.DARK, bordercolor=self.DARK,
                        arrowcolor=self.MUTED, relief="flat")
        style.configure("Horizontal.TScrollbar", background=self.CARD,
                        troughcolor=self.DARK, bordercolor=self.DARK,
                        arrowcolor=self.MUTED, relief="flat")
        style.map("Vertical.TScrollbar", background=[("active", self.ACCENT)])
        style.map("Horizontal.TScrollbar", background=[("active", self.ACCENT)])
        style.configure("TCombobox", fieldbackground=self.CARD,
                        background=self.CARD, foreground=self.TEXT,
                        arrowcolor=self.ACCENT, borderwidth=0)
        style.map("TCombobox", fieldbackground=[("readonly", self.CARD)],
                 foreground=[("readonly", self.TEXT)])
        self.option_add("*TCombobox*Listbox.background", self.CARD)
        self.option_add("*TCombobox*Listbox.foreground", self.TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", self.ACCENT)

    # ── UI construction ──────────────────────────────────────────────────────
    def _build_ui(self):
        conn = tk.Frame(self, bg=self.PANEL, bd=0)
        conn.pack(fill="x", padx=12, pady=(12, 4))

        tk.Label(conn, text="VEHICLE IP", bg=self.PANEL, fg=self.MUTED,
                 font=("Consolas", 8)).grid(row=0, column=0, sticky="w",
                                            padx=(8, 2), pady=6)
        self.host_var = tk.StringVar(value="192.168.1.1")
        tk.Entry(conn, textvariable=self.host_var, width=15,
                 bg=self.CARD, fg=self.TEXT, insertbackground=self.ACCENT,
                 relief="flat", font=self.FONT, bd=4).grid(row=0, column=1,
                                                           padx=4)
        tk.Label(conn, text="PORT", bg=self.PANEL, fg=self.MUTED,
                 font=("Consolas", 8)).grid(row=0, column=2, sticky="w",
                                            padx=(8, 2))
        self.port_var = tk.StringVar(value=str(TCP_DEFAULT_PORT))
        tk.Entry(conn, textvariable=self.port_var, width=6,
                 bg=self.CARD, fg=self.TEXT, insertbackground=self.ACCENT,
                 relief="flat", font=self.FONT, bd=4).grid(row=0, column=3,
                                                           padx=4)
        self.conn_btn = tk.Button(conn, text="CONNECT",
                                  command=self._toggle_connect,
                                  bg=self.GREEN, fg=self.DARK, relief="flat",
                                  font=("Consolas", 9, "bold"), cursor="hand2",
                                  padx=14, pady=4, bd=0,
                                  activebackground="#00c853")
        self.conn_btn.grid(row=0, column=4, padx=(10, 8))
        self.conn_indicator = tk.Canvas(conn, width=12, height=12,
                                        bg=self.PANEL, highlightthickness=0)
        self.conn_indicator.grid(row=0, column=5, padx=(0, 8))
        self._draw_indicator(False)

        container = tk.Frame(self, bg=self.DARK)
        container.pack(fill="both", expand=True, padx=12, pady=(4, 12))

        main_area = tk.Frame(container, bg=self.DARK)
        main_area.pack(side="left", fill="both", expand=True, padx=(0, 8))
        tk.Label(main_area, text="  STATE 1 · LANE TRACKING", bg=self.PANEL,
                 fg=self.ACCENT, font=("Consolas", 9, "bold"), anchor="w"
                 ).pack(fill="x", pady=(0, 4))

        sidebar_outer = tk.Frame(container, bg=self.DARK, width=360)
        sidebar_outer.pack(side="right", fill="y")
        sidebar_outer.pack_propagate(False)
        tk.Label(sidebar_outer, text="  STATE 2/3 · APRILTAG + GRIPPER (always on)",
                 bg=self.PANEL, fg=self.ACCENT, font=("Consolas", 9, "bold"),
                 anchor="w").pack(fill="x", pady=(0, 4))

        self._build_lane_tab(main_area)
        self._build_mission_sidebar(sidebar_outer)

    def _build_lane_tab(self, parent):
        # Two-pane layout: a settings NOTEBOOK on the left (Drive & Path /
        # Detection & PID / Road Type - click a tab, everything's there) and
        # an ALWAYS-VISIBLE pane on the right (joystick+telemetry aren't
        # needed simultaneously with tuning, but the camera preview, lane
        # status, keymap and log are things you want in sight no matter
        # which tab is open).
        body = tk.Frame(parent, bg=self.DARK)
        body.pack(fill="both", expand=True)

        notebook_col = tk.Frame(body, bg=self.DARK)
        notebook_col.pack(side="left", fill="both", expand=False, padx=(0, 8))

        self.notebook = ttk.Notebook(notebook_col)
        self.notebook.pack(fill="both", expand=True)

        tab_drive = tk.Frame(self.notebook, bg=self.DARK)
        tab_detect = tk.Frame(self.notebook, bg=self.DARK)
        tab_road = tk.Frame(self.notebook, bg=self.DARK)
        self.notebook.add(tab_drive, text="  DRIVE & PATH  ")
        self.notebook.add(tab_detect, text="  DETECTION & PID  ")
        self.notebook.add(tab_road, text="  ROAD TYPE  ")

        self._build_joystick(tab_drive)
        self._build_gauges(tab_drive)
        self._build_servo_settings(tab_drive)
        self._build_path_controls(tab_drive)

        self._build_lane_keep_controls(tab_detect)

        self._build_road_type_controls(tab_road)
        auto_card = self._card(tab_road, "  START / STOP")
        self.auto_btn = tk.Button(auto_card, text="START AUTO (L)",
                                  command=self._toggle_autonomous,
                                  bg=self.GREEN, fg=self.DARK, relief="flat",
                                  font=("Consolas", 10, "bold"), cursor="hand2",
                                  padx=8, pady=6, bd=0,
                                  activebackground="#00c853")
        self.auto_btn.pack(fill="x", pady=(0, 6))
        btn_row = tk.Frame(auto_card, bg=self.CARD)
        btn_row.pack(fill="x")
        tk.Button(btn_row, text="CALIBRATE", command=self._calibrate_camera,
                  bg=self.ACCENT, fg=self.DARK, relief="flat",
                  font=("Consolas", 9, "bold"), cursor="hand2", padx=8,
                  pady=4, bd=0, activebackground="#00b8e0"
                  ).pack(side="left", fill="x", expand=True, padx=(0, 4))
        tk.Button(btn_row, text="SCAN CAMS", command=self._scan_cameras_ui,
                  bg=self.MUTED, fg=self.DARK, relief="flat",
                  font=("Consolas", 9, "bold"), cursor="hand2", padx=8,
                  pady=4, bd=0, activebackground="#4a5a6a"
                  ).pack(side="left", fill="x", expand=True)

        right = tk.Frame(body, bg=self.DARK)
        right.pack(side="left", fill="both", expand=True)
        self._build_preview_and_confirm(right)
        self._build_keymap(right)
        self._build_log(right)

    # ── STATE 2 / STATE 3 side panel (AprilTag centering + gripper) ────────
    def _build_mission_sidebar(self, parent):
        outer = tk.Frame(parent, bg=self.DARK)
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, bg=self.DARK, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        col = tk.Frame(canvas, bg=self.DARK)
        col_window = canvas.create_window((0, 0), window=col, anchor="nw")

        def _on_col_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        col.bind("<Configure>", _on_col_configure)

        def _on_canvas_configure(event):
            canvas.itemconfigure(col_window, width=event.width)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            delta = -1 if event.delta > 0 else 1
            canvas.yview_scroll(delta, "units")
        canvas.bind("<Enter>", lambda e: (
            canvas.bind_all("<MouseWheel>", _on_mousewheel),
            canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units")),
            canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))))
        canvas.bind("<Leave>", lambda e: (
            canvas.unbind_all("<MouseWheel>"),
            canvas.unbind_all("<Button-4>"),
            canvas.unbind_all("<Button-5>")))

        status = tk.Frame(col, bg=self.PANEL)
        status.pack(fill="x", pady=(0, 6))
        self.mission_status_lbl = tk.Label(
            status, text="IDLE - press START AUTO", wraplength=330,
            bg=self.PANEL, fg=self.YELLOW, font=("Consolas", 10, "bold"), justify="center")
        self.mission_status_lbl.pack(pady=8)

        stream_card = self._card(col, "  APRILTAG CAMERA STREAM (:8080)")
        self.mission_canvas = tk.Canvas(stream_card, width=330, height=250,
                                        bg="#0d1117", highlightthickness=1,
                                        highlightbackground=self.MUTED)
        self.mission_canvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.mission_canvas.create_text(
            165, 125, text="waiting for stream...", fill=self.MUTED,
            font=("Consolas", 9), justify="center", tags="placeholder")

        conn_card = self._card(col, "  STREAM CONNECTION")
        self.mission_ip_var = tk.StringVar(value=ESP32_CAM_IP)
        self.mission_port_var = tk.StringVar(value=str(ESP32_CAM_PORT))
        self._entry_row_var(conn_card, "Cam IP", self.mission_ip_var)
        self._entry_row_var(conn_card, "Cam port", self.mission_port_var, width=6)
        tk.Button(conn_card, text="RECONNECT STREAM", command=self._on_mission_reconnect,
                  bg=self.ACCENT, fg=self.DARK, relief="flat",
                  font=("Consolas", 9, "bold"), cursor="hand2", padx=8, pady=4,
                  bd=0, activebackground="#00b8e0").pack(fill="x", padx=8, pady=(4, 8))

        res_card = self._card(col, "  STREAM RESOLUTION (adjustable)")
        self.mission_res_var = tk.StringVar(value="SVGA 800x600")
        ttk.Combobox(res_card, textvariable=self.mission_res_var, state="readonly",
                    values=list(FRAMESIZE_PRESETS.keys())).pack(fill="x", padx=8, pady=(0, 4))
        tk.Button(res_card, text="APPLY RESOLUTION", command=self._on_mission_apply_resolution,
                  bg=self.ACCENT, fg=self.DARK, relief="flat",
                  font=("Consolas", 9, "bold"), cursor="hand2", padx=8, pady=4,
                  bd=0, activebackground="#00b8e0").pack(fill="x", padx=8, pady=(0, 8))

        pose_card = self._card(col, "  TAG / POSE")
        self.mission_pose_lbl = tk.Label(
            pose_card, text="Distance  : -\nLeft/Right: -\nYaw       : -\nMargin    : -",
            bg=self.CARD, fg="#7ce0c8", font=("Consolas", 9), justify="left", anchor="w")
        self.mission_pose_lbl.pack(fill="x", padx=8, pady=(0, 8))

        pid_card = self._card(col, "  APRILTAG PID (STATE 2 · CENTERING)")
        self.tag_kp_var = self._entry_row(pid_card, "Kp", str(TAG_PID_KP), 6)
        self.tag_ki_var = self._entry_row(pid_card, "Ki", str(TAG_PID_KI), 6)
        self.tag_kd_var = self._entry_row(pid_card, "Kd", str(TAG_PID_KD), 6)
        self.tag_margin_var = self._entry_row(pid_card, "Error mm", str(TAG_ERROR_MARGIN_MM), 6)
        tk.Button(pid_card, text="APPLY PID", command=self._on_mission_apply_pid,
                  bg=self.ACCENT, fg=self.DARK, relief="flat",
                  font=("Consolas", 9, "bold"), cursor="hand2", padx=8, pady=4,
                  bd=0, activebackground="#00b8e0").pack(fill="x", padx=8, pady=(4, 8))

        center_card = self._card(col, "  CENTERING DEBUG (STATE 2a)")
        self.mission_center_lbl = tk.Label(
            center_card, text="Pixel err : -\nP/I/D     : -\nOutput mm : -\nCmd deg   : -",
            bg=self.CARD, fg="#e0c87c", font=("Consolas", 9), justify="left", anchor="w")
        self.mission_center_lbl.pack(fill="x", padx=8, pady=(0, 8))

        grip_tune_card = self._card(col, "  GRIPPER TUNING (STATE 3)")
        self.grip_step_var = self._entry_row(grip_tune_card, "Step size (deg)",
                                             str(GRIPPER_RAMP_STEP), 6)
        self.grip_step_time_var = self._entry_row(grip_tune_card, "Step time (s)",
                                                   str(GRIPPER_RAMP_INTERVAL), 6)
        self.grip_pause_var = self._entry_row(grip_tune_card, "Pause time (s)",
                                              str(GRIPPER_HOLD_AT_MAX_SEC), 6)
        tk.Button(grip_tune_card, text="APPLY GRIPPER PARAMS",
                  command=self._on_mission_apply_gripper,
                  bg=self.ACCENT, fg=self.DARK, relief="flat",
                  font=("Consolas", 9, "bold"), cursor="hand2", padx=8, pady=4,
                  bd=0, activebackground="#00b8e0").pack(fill="x", padx=8, pady=(4, 8))

        grip_dbg_card = self._card(col, "  GRIPPER DEBUG (STATE 3)")
        self.mission_grip_lbl = tk.Label(
            grip_dbg_card, text="Phase : -\nSET3  : -\nENC3  : -",
            bg=self.CARD, fg="#f0a060", font=("Consolas", 9), justify="left", anchor="w")
        self.mission_grip_lbl.pack(fill="x", padx=8, pady=(0, 8))

        note_card = self._card(col, "  STOP SIGN (PC CAMERA)")
        tk.Label(note_card,
                 text="Stop-sign detection on the PC camera stream is NOT "
                      "YET IMPLEMENTED (stub always reports 'not seen'). "
                      "See detect_stop_sign() in the code.",
                 bg=self.CARD, fg=self.MUTED, font=("Consolas", 8),
                 wraplength=310, justify="left").pack(fill="x", padx=8, pady=(0, 8))

        ctrl_card = self._card(col, "  MISSION CONTROL")
        tk.Button(ctrl_card, text="⏹  E-STOP", command=self._on_mission_estop,
                  bg=self.RED, fg=self.DARK, relief="flat",
                  font=("Consolas", 10, "bold"), cursor="hand2", padx=8, pady=4,
                  bd=0, activebackground="#cc3333").pack(fill="x", padx=8, pady=(6, 2))
        tk.Button(ctrl_card, text="⟳  Force Align (State 2)", command=self._on_mission_align,
                  bg="#5a1a8a", fg="white", relief="flat",
                  font=("Consolas", 9, "bold"), cursor="hand2", padx=8, pady=4,
                  bd=0, activebackground="#421268").pack(fill="x", padx=8, pady=2)
        tk.Button(ctrl_card, text="✊  Force Grip (State 3)", command=self._on_mission_grip,
                  bg="#8a4a1a", fg="white", relief="flat",
                  font=("Consolas", 9, "bold"), cursor="hand2", padx=8, pady=4,
                  bd=0, activebackground="#6b3712").pack(fill="x", padx=8, pady=2)
        self.mission_reset_btn = tk.Button(
            ctrl_card, text="✔  RESET & REPEAT", command=self._on_mission_reset_repeat,
            bg=self.GREEN, fg=self.DARK, relief="flat",
            font=("Consolas", 10, "bold"), cursor="hand2", padx=8, pady=4,
            bd=0, activebackground="#00c853")
        self.mission_reset_btn.pack(fill="x", padx=8, pady=(2, 8))
        tk.Label(ctrl_card, text="RESET & REPEAT sends \"RESET\" to port 4545 "
                                 "and restarts STATE 1 lane following. It "
                                 "lights up once STATE 3 finishes, but can "
                                 "also be used any time as a manual override.",
                 bg=self.CARD, fg=self.MUTED, font=("Consolas", 7),
                 wraplength=310, justify="left").pack(fill="x", padx=8, pady=(0, 10))

    def _entry_row_var(self, parent, label, var, width=14):
        row = tk.Frame(parent, bg=self.CARD)
        row.pack(fill="x", padx=8, pady=2)
        tk.Label(row, text=label, bg=self.CARD, fg=self.TEXT,
                 font=("Consolas", 8), width=8, anchor="w").pack(side="left")
        tk.Entry(row, textvariable=var, width=width, bg=self.DARK, fg=self.TEXT,
                 insertbackground=self.ACCENT, relief="flat", bd=2
                 ).pack(side="left", fill="x", expand=True)
        return var

    def _card(self, parent, title):
        f = tk.Frame(parent, bg=self.CARD, bd=0)
        f.pack(fill="x", pady=4)
        tk.Label(f, text=title, bg=self.CARD, fg=self.ACCENT,
                 font=("Consolas", 8, "bold")).pack(anchor="w", padx=10,
                                                    pady=(8, 2))
        inner = tk.Frame(f, bg=self.CARD)
        inner.pack(fill="x", padx=10, pady=(0, 8))
        return inner

    def _entry_row(self, parent, label, default, width=6, hint=""):
        row = tk.Frame(parent, bg=self.CARD)
        row.pack(fill="x", pady=1)
        tk.Label(row, text=label, bg=self.CARD, fg=self.TEXT,
                 font=("Consolas", 9), width=13, anchor="w").pack(side="left")
        var = tk.StringVar(value=str(default))
        tk.Entry(row, textvariable=var, width=width, bg="#0d1117",
                 fg=self.ACCENT, insertbackground=self.ACCENT, relief="flat",
                 font=("Consolas", 10, "bold"), bd=3).pack(side="left", padx=4)
        if hint:
            tk.Label(row, text=hint, bg=self.CARD, fg=self.MUTED,
                     font=("Consolas", 7)).pack(side="left", padx=4)
        return var

    def _scale_row(self, parent, label, frm, to, default, res=1):
        row = tk.Frame(parent, bg=self.CARD)
        row.pack(fill="x", pady=1)
        tk.Label(row, text=label, bg=self.CARD, fg=self.TEXT,
                 font=("Consolas", 9), width=13, anchor="w").pack(side="left")
        var = tk.DoubleVar(value=default)
        tk.Scale(row, variable=var, from_=frm, to=to, resolution=res,
                 orient="horizontal", length=150, bg=self.CARD, fg=self.ACCENT,
                 troughcolor="#0d1117", highlightthickness=0, bd=0,
                 font=("Consolas", 7), activebackground=self.ACCENT
                 ).pack(side="left", padx=4)
        return var

    def _build_joystick(self, parent):
        inner = self._card(parent, "  JOYSTICK")
        self.joy_canvas = tk.Canvas(inner, width=170, height=170,
                                    bg="#0d1117", highlightthickness=1,
                                    highlightbackground=self.MUTED)
        self.joy_canvas.pack()
        self._draw_joystick(0, 0)

    def _draw_joystick(self, norm_x, norm_y):
        c = self.joy_canvas
        c.delete("all")
        cx, cy, r = 85, 85, 70
        for ri in (23, 46, 70):
            c.create_oval(cx - ri, cy - ri, cx + ri, cy + ri,
                          outline=self.MUTED, width=1)
        c.create_line(cx - r, cy, cx + r, cy, fill=self.MUTED)
        c.create_line(cx, cy - r, cx, cy + r, fill=self.MUTED)
        dx = cx + int(norm_x * r)
        dy = cy - int(norm_y * r)
        c.create_oval(dx - 9, dy - 9, dx + 9, dy + 9,
                      fill=self.ACCENT, outline="white", width=2)

    def _build_gauges(self, parent):
        inner = self._card(parent, "  TELEMETRY")
        tk.Label(inner, text="SPEED", bg=self.CARD, fg=self.MUTED,
                 font=("Consolas", 8)).grid(row=0, column=0, sticky="w")
        self.speed_lbl = tk.Label(inner, text="   0", bg=self.CARD,
                                  fg=self.ACCENT,
                                  font=("Consolas", 10, "bold"), width=6)
        self.speed_lbl.grid(row=0, column=1, sticky="e")
        self.speed_bar = tk.Canvas(inner, width=160, height=12, bg="#0d1117",
                                   highlightthickness=0)
        self.speed_bar.grid(row=1, column=0, columnspan=2, pady=(2, 6),
                            sticky="w")
        tk.Label(inner, text="STEER", bg=self.CARD, fg=self.MUTED,
                 font=("Consolas", 8)).grid(row=2, column=0, sticky="w")
        self.steer_lbl = tk.Label(inner, text=str(STEER_CENTER), bg=self.CARD,
                                  fg=self.ACCENT,
                                  font=("Consolas", 10, "bold"), width=6)
        self.steer_lbl.grid(row=2, column=1, sticky="e")
        self.steer_bar = tk.Canvas(inner, width=160, height=12, bg="#0d1117",
                                   highlightthickness=0)
        self.steer_bar.grid(row=3, column=0, columnspan=2, pady=(2, 4),
                            sticky="w")
        self._update_bars(0, STEER_CENTER)

    def _update_bars(self, speed, steer):
        sb = self.speed_bar
        sb.delete("all")
        wpx = int(abs(speed) / SPEED_MAX_VALUE * 78)
        color = self.RED if speed < 0 else self.GREEN
        if speed < 0:
            sb.create_rectangle(80 - wpx, 0, 80, 12, fill=color, outline="")
        else:
            sb.create_rectangle(80, 0, 80 + wpx, 12, fill=color, outline="")
        sb.create_line(80, 0, 80, 12, fill=self.MUTED, width=2)

        stb = self.steer_bar
        stb.delete("all")
        lo, hi = self.ctrl.steer_min, self.ctrl.steer_max
        norm_s = (steer - lo) / max(hi - lo, 1)
        dot_x = int(norm_s * 160)
        stb.create_rectangle(0, 4, 160, 8, fill="#1e2230", outline="")
        stb.create_rectangle(dot_x - 6, 0, dot_x + 6, 12, fill=self.ACCENT,
                             outline="")

    def _build_servo_settings(self, parent):
        inner = self._card(parent, "  SERVO SETTINGS")
        self.center_var = self._entry_row(inner, "Center",
                                          self.ctrl.steer_center)
        self.min_var = self._entry_row(inner, "Min (right)",
                                       self.ctrl.steer_min)
        self.max_var = self._entry_row(inner, "Max (left)",
                                       self.ctrl.steer_max)
        tk.Button(inner, text="APPLY & SEND", command=self._apply_servo,
                  bg=self.ACCENT, fg=self.DARK, relief="flat",
                  font=("Consolas", 9, "bold"), cursor="hand2", padx=12,
                  pady=4, bd=0, activebackground="#00b8e0"
                  ).pack(fill="x", pady=(6, 0))

    def _build_path_controls(self, parent):
        inner = self._card(parent, "  PATH RECORD / PLAYBACK")
        self.path_status_lbl = tk.Label(inner, text="No path loaded.",
                                        bg=self.CARD, fg=self.MUTED,
                                        font=("Consolas", 8), wraplength=190,
                                        justify="left", anchor="w")
        self.path_status_lbl.pack(fill="x", pady=(0, 6))
        btn_row = tk.Frame(inner, bg=self.CARD)
        btn_row.pack(fill="x")
        tk.Button(btn_row, text="LOAD FILE", command=self._load_path_dialog,
                  bg=self.ACCENT, fg=self.DARK, relief="flat",
                  font=("Consolas", 9, "bold"), cursor="hand2", padx=8,
                  pady=4, bd=0, activebackground="#00b8e0"
                  ).pack(side="left", padx=(0, 6))
        tk.Button(btn_row, text="PLAY (P)", command=self.ctrl.play_path,
                  bg=self.GREEN, fg=self.DARK, relief="flat",
                  font=("Consolas", 9, "bold"), cursor="hand2", padx=8,
                  pady=4, bd=0, activebackground="#00c853").pack(side="left")

    def _build_road_type_controls(self, parent):
        inner = self._card(parent, "  STREET TYPE")
        tk.Label(inner, text="Select the road's color palette before "
                             "starting lane tracking:",
                 bg=self.CARD, fg=self.MUTED, font=("Consolas", 8),
                 wraplength=260, justify="left").pack(anchor="w",
                                                      pady=(0, 4))
        self.road_type_var = tk.StringVar(value=LaneKeepController.ROAD_TYPE_2_BLACK)
        for label, val in (("2 BLACK BORDERS  (white road, black edges "
                            "+ optional yellow divider)",
                            LaneKeepController.ROAD_TYPE_2_BLACK),
                           ("YELLOW FOLLOWER  (single yellow line, keep "
                            "it inside a region)",
                            LaneKeepController.ROAD_TYPE_YELLOW)):
            tk.Radiobutton(inner, text=label, value=val,
                          variable=self.road_type_var,
                          command=self._on_road_type_change,
                          bg=self.CARD, fg=self.TEXT, selectcolor=self.DARK,
                          activebackground=self.CARD, wraplength=260,
                          justify="left", font=("Consolas", 8)
                          ).pack(anchor="w", pady=1)

        self.yellow_region_frame = tk.Frame(inner, bg=self.CARD)
        tk.Label(self.yellow_region_frame, text="── target region ──",
                 bg=self.CARD, fg=self.MUTED, font=("Consolas", 7)
                 ).pack(fill="x", pady=(4, 0))
        self.yf_width_var = self._scale_row(
            self.yellow_region_frame, "Region width %", 5, 100, 40)
        self.yf_offset_var = self._scale_row(
            self.yellow_region_frame, "Region offset %", -100, 100, 0)
        tk.Label(self.yellow_region_frame,
                 text="Width/position of the band the yellow line must "
                      "stay inside (auto-clamped to the camera view). "
                      "Shown on the preview as two GREEN vertical lines.",
                 bg=self.CARD, fg=self.MUTED, font=("Consolas", 7),
                 wraplength=260, justify="left").pack(anchor="w", pady=(2, 0))
        self._on_road_type_change()

    def _on_road_type_change(self):
        if self.road_type_var.get() == LaneKeepController.ROAD_TYPE_YELLOW:
            self.yellow_region_frame.pack(fill="x")
        else:
            self.yellow_region_frame.pack_forget()

    def _build_lane_keep_controls(self, parent):
        inner = self._card(parent, "  MULTI-LANE TRACKING")

        self.cam_index_var = self._entry_row(inner, "Camera idx", "0", 4)
        self.scan_height_var = self._scale_row(
            inner, "Scan line %", 20, 90, 55)
        self.black_thresh_var = self._scale_row(
            inner, "Black thresh", 10, 200, 70)
        self.num_rows_var = self._scale_row(
            inner, "Resolution", 6, 36, 14)
        self.lpf_alpha_var = self._scale_row(
            inner, "LPF alpha", 0.05, 1.0, 0.7, res=0.05)
        self.min_w_var = self._entry_row(inner, "Min line px", "5", 4)
        self.max_w_var = self._entry_row(inner, "Max line px", "110", 4)

        tk.Label(inner, text="── yellow divider ──", bg=self.CARD,
                 fg=self.MUTED, font=("Consolas", 7)).pack(fill="x",
                                                            pady=(4, 0))
        self.yellow_sat_var = self._scale_row(
            inner, "Yellow sat min", 0, 255, 80)
        self.yellow_val_var = self._scale_row(
            inner, "Yellow val min", 0, 255, 90)

        tk.Label(inner, text="── lane memory (home-lane tracking) ──",
                 bg=self.CARD, fg=self.MUTED, font=("Consolas", 7)
                 ).pack(fill="x", pady=(4, 0))
        self.memory_frames_var = self._scale_row(
            inner, "Memory frames", 5, 120, 25)
        tk.Label(inner,
                 text="How many frames a lost border line is remembered "
                      "for (shifted with the rest, confidence decaying) "
                      "before being forgotten. The dedicated HOME LANE "
                      "width (learned once both borders of YOUR lane are "
                      "seen together) is used ahead of this to re-draw a "
                      "missing border and to steer back if both are lost.",
                 bg=self.CARD, fg=self.MUTED, font=("Consolas", 7),
                 wraplength=260, justify="left").pack(anchor="w", pady=(2, 4))

        tk.Label(inner, text="── PID / control ──", bg=self.CARD,
                 fg=self.MUTED, font=("Consolas", 7)).pack(fill="x",
                                                            pady=(4, 0))
        self.kp_var = self._entry_row(inner, "PID kp", "1.6", 5)
        self.ki_var = self._entry_row(inner, "PID ki", "0.05", 5,
                                      "(anti wind-up on)")
        self.kd_var = self._entry_row(inner, "PID kd", "0.35", 5)
        self.deadband_var = self._entry_row(inner, "Error margin", "0.04", 5,
                                            "(deadband, 0..1)")
        self.calib_margin_var = self._entry_row(inner, "Calib margin", "0.12",
                                                5, "(frac of lane width)")
        self.cruise_speed_var = self._entry_row(inner, "Cruise speed", "350",
                                                5, "(0-1023)")
        self.slowdown_var = self._entry_row(inner, "Turn slowdown", "0.5", 5,
                                            "(0-0.9)")
        self.target_offset_var = self._scale_row(
            inner, "Target offset", -100, 100, 0)

        checks = tk.Frame(inner, bg=self.CARD)
        checks.pack(fill="x", pady=(2, 4))
        self.flip_var = tk.BooleanVar(value=False)
        tk.Checkbutton(checks, text="Flip cam", variable=self.flip_var,
                       bg=self.CARD, fg=self.TEXT, selectcolor=self.DARK,
                       activebackground=self.CARD, font=("Consolas", 8)
                       ).pack(side="left")
        self.invert_var = tk.BooleanVar(value=False)
        tk.Checkbutton(checks, text="Invert steering",
                       variable=self.invert_var, bg=self.CARD, fg=self.TEXT,
                       selectcolor=self.DARK, activebackground=self.CARD,
                       font=("Consolas", 8)).pack(side="left", padx=8)

    def _build_preview_and_confirm(self, parent):
        inner = self._card(parent, "  CAMERA / LANE VIEW")
        self.camera_canvas = tk.Canvas(inner, width=300, height=225,
                                       bg="#0d1117", highlightthickness=1,
                                       highlightbackground=self.MUTED)
        self.camera_canvas.pack()
        self.camera_canvas.create_text(150, 112,
                                       text="camera preview\n(starts with AUTO)",
                                       fill=self.MUTED, font=("Consolas", 9),
                                       justify="center")
        self.lane_status_lbl = tk.Label(inner, text="lane: -", bg=self.CARD,
                                        fg=self.YELLOW, font=("Consolas", 8),
                                        anchor="w")
        self.lane_status_lbl.pack(fill="x", pady=(4, 0))

        self.confirm_frame = tk.Frame(inner, bg="#332b12")
        self.confirm_lbl = tk.Label(self.confirm_frame, text="",
                                    bg="#332b12", fg=self.YELLOW,
                                    font=("Consolas", 8), wraplength=290,
                                    justify="left")
        self.confirm_lbl.pack(fill="x", padx=6, pady=(6, 2))
        brow = tk.Frame(self.confirm_frame, bg="#332b12")
        brow.pack(pady=(0, 6))
        for txt, shift, col in (("< SHIFT LEFT", -1, self.ACCENT),
                                ("CORRECT ✓", 0, self.GREEN),
                                ("SHIFT RIGHT >", +1, self.ACCENT)):
            tk.Button(brow, text=txt, bg=col, fg=self.DARK, relief="flat",
                      font=("Consolas", 8, "bold"), cursor="hand2", padx=6,
                      pady=3, bd=0,
                      command=lambda s=shift: self._confirm_lane(s)
                      ).pack(side="left", padx=3)

    def _build_keymap(self, parent):
        inner = self._card(parent, "  KEYBOARD MAP")
        for label, desc in ((" W ", "Accelerate forward"),
                            (" S ", "Brake / reverse"),
                            ("A/D", "Steer left / right"),
                            (" R ", "Start/stop recording path"),
                            (" P ", "Play back recorded path"),
                            (" L ", "Toggle lane tracking")):
            cell = tk.Frame(inner, bg=self.CARD)
            cell.pack(anchor="w", pady=1)
            tk.Label(cell, text=label, bg="#252a35", fg=self.ACCENT,
                     font=("Consolas", 9, "bold"), width=3, relief="solid",
                     bd=1).pack(side="left", padx=(0, 6))
            tk.Label(cell, text=desc, bg=self.CARD, fg=self.MUTED,
                     font=("Consolas", 8)).pack(side="left")

    def _build_log(self, parent):
        inner = self._card(parent, "  COMMAND LOG")
        self.log_box = tk.Text(inner, width=40, height=9, bg="#0d1117",
                               fg="#8899aa", font=("Consolas", 8),
                               relief="flat", state="disabled", bd=0)
        self.log_box.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(inner, command=self.log_box.yview)
        sb.pack(side="right", fill="y")
        self.log_box.configure(yscrollcommand=sb.set)

     # ── Key bindings ─────────────────────────────────────────────────────────
    def _is_typing_target(self):
        w = self.focus_get()
        return isinstance(w, (tk.Entry, tk.Text))

    def _guarded(self, fn):
        if not self._is_typing_target():
            fn()

    def _make_key_down(self, k):
        def handler(e):
            if not self._is_typing_target():
                self.ctrl.key_down(k)
        return handler

    def _make_key_up(self, k):
        def handler(e):
            if not self._is_typing_target():
                self.ctrl.key_up(k)
        return handler
    def _reclaim_focus(self, event):
    # Clicking anywhere that ISN'T a text entry/box sends keyboard
    # focus back to the main window, so WASD always drives the
    # vehicle instead of leaking into whatever box was last clicked.
       if not isinstance(event.widget, (tk.Entry, tk.Text)):
        self.focus_set()
    def _bind_keys(self):
        for k in ("w", "s", "a", "d", "W", "S", "A", "D"):
            self.bind(f"<KeyPress-{k}>", self._make_key_down(k.lower()))
            self.bind(f"<KeyRelease-{k}>", self._make_key_up(k.lower()))
        self.bind("<KeyPress-r>", lambda e: self._guarded(self._toggle_recording_ui))
        self.bind("<KeyPress-R>", lambda e: self._guarded(self._toggle_recording_ui))
        self.bind("<KeyPress-p>", lambda e: self._guarded(self.ctrl.play_path))
        self.bind("<KeyPress-P>", lambda e: self._guarded(self.ctrl.play_path))
        self.bind("<KeyPress-l>", lambda e: self._guarded(self._toggle_autonomous))
        self.bind("<KeyPress-L>", lambda e: self._guarded(self._toggle_autonomous))
        self.bind_all("<Button-1>", self._reclaim_focus)   # <-- new line


    # ── Live parameter snapshot (GUI thread -> worker thread, lock-guarded) ──
    def _snapshot_params(self):
        def f(var, default):
            try:
                return float(var.get())
            except (ValueError, tk.TclError):
                return default
        snap = {
            "black_thresh":   int(f(self.black_thresh_var, 70)),
            "roi_top_ratio":  f(self.scan_height_var, 55) / 100.0,
            "num_rows":       int(f(self.num_rows_var, 14)),
            "lpf_alpha":      f(self.lpf_alpha_var, 0.7),
            "min_line_w":     int(f(self.min_w_var, 5)),
            "max_line_w":     int(f(self.max_w_var, 110)),
            "yellow_sat_min": int(f(self.yellow_sat_var, 80)),
            "yellow_val_min": int(f(self.yellow_val_var, 90)),
            "memory_frames":  int(f(self.memory_frames_var, 25)),
            "kp":             f(self.kp_var, 1.6),
            "ki":             f(self.ki_var, 0.05),
            "kd":             f(self.kd_var, 0.35),
            "deadband":       f(self.deadband_var, 0.04),
            "calib_margin":   f(self.calib_margin_var, 0.12),
            "cruise_speed":   int(f(self.cruise_speed_var, 350)),
            "max_turn_slowdown": f(self.slowdown_var, 0.5),
            "target_offset":  f(self.target_offset_var, 0) / 100.0,
            "flip_horizontal": bool(self.flip_var.get()),
            "invert_steering": bool(self.invert_var.get()),
            "yf_region_width_pct":  f(self.yf_width_var, 40),
            "yf_region_offset_pct": f(self.yf_offset_var, 0),
        }
        with self._params_lock:
            self._lk_params = snap
        self.after(100, self._snapshot_params)

    def _get_lk_params(self):
        with self._params_lock:
            return dict(self._lk_params)

    # ── Path UI helpers ──────────────────────────────────────────────────────
    def _load_path_dialog(self):
        filename = filedialog.askopenfilename(
            title="Select recorded path file",
            filetypes=[("Path text files", "*.txt"), ("All files", "*.*")])
        if not filename:
            return
        if self.ctrl.load_path_file(filename):
            self.path_status_lbl.config(
                text=f"Loaded: {os.path.basename(filename)}\n"
                     f"({len(self.ctrl.active_path)} pts) - press P")

    def _toggle_recording_ui(self):
        was_recording = self.ctrl.recording
        self.ctrl.toggle_recording()
        if was_recording and not self.ctrl.recording and self.ctrl.active_source:
            self.path_status_lbl.config(
                text=f"Recorded: {self.ctrl.active_source}\n"
                     f"({len(self.ctrl.active_path)} pts) - press P")
        elif not was_recording and self.ctrl.recording:
            self.path_status_lbl.config(text="Recording...")

    # ── Camera scan / calibrate ──────────────────────────────────────────────
    def _scan_cameras_ui(self):
        if not CV2_AVAILABLE:
            messagebox.showerror("Missing dependency",
                                 "pip install opencv-python numpy")
            return
        if self.ctrl.autonomous:
            messagebox.showinfo("Busy", "Stop lane tracking (L) first - it's "
                                        "already using a camera.")
            return
        self._log("Scanning camera indices 0-5 (can take a few seconds)...")
        threading.Thread(target=self._scan_cameras_worker, daemon=True).start()

    def _scan_cameras_worker(self):
        working = scan_cameras(max_index=6, log_cb=self._log)
        if working:
            best = working[0]
            self._log(f"Found {len(working)} camera(s). Using index "
                      f"{best['index']} ({best['backend']}, "
                      f"{best['width']}x{best['height']}).")
            self.after(0, self.cam_index_var.set, str(best["index"]))

    def _calibrate_camera(self):
        if not CV2_AVAILABLE:
            messagebox.showerror("Missing dependency",
                                 "pip install opencv-python numpy")
            return
        if self.ctrl.autonomous:
            messagebox.showinfo("Busy", "Stop lane tracking (L) before "
                                        "calibrating.")
            return
        try:
            cam_idx = int(float(self.cam_index_var.get()))
        except ValueError:
            messagebox.showerror("Invalid", "Camera index must be an integer.")
            return
        threading.Thread(target=self._calibrate_worker, args=(cam_idx,),
                         daemon=True).start()

    def _calibrate_worker(self, cam_idx):
        cap = open_camera(cam_idx, self._log)
        if cap is None:
            self._log(f"Could not open camera index {cam_idx}.")
            return
        frame = None
        for _ in range(6):
            ok, f = cap.read()
            if ok:
                frame = f
            time.sleep(0.03)
        cap.release()
        if frame is None:
            self._log("Could not read a frame from the camera.")
            return
        if frame.std() < 3:
            self._log("Camera returned a flat/uniform frame - the source "
                      "isn't streaming a real picture yet. Try again.")
            self._on_camera_frame(frame, {})
            return

        frame = cv2.resize(frame, (400, 300))
        params = self._get_lk_params()
        if params.get("flip_horizontal"):
            frame = cv2.flip(frame, 1)

        det = LaneDetector()
        det.roi_top_ratio = params["roi_top_ratio"]
        det.min_line_w = params["min_line_w"]
        det.max_line_w = params["max_line_w"]
        det.num_rows = params.get("num_rows", 14)
        det.lpf_alpha = 1.0
        det.yellow_sat_min = params.get("yellow_sat_min", det.yellow_sat_min)
        det.yellow_val_min = params.get("yellow_val_min", det.yellow_val_min)
        otsu, stats = det.suggest_threshold(frame)
        det.black_thresh = int(otsu)
        result = det.process(frame)

        self._log(f"Calibration cam {cam_idx}: ROI y={stats['roi_top']}-"
                  f"{stats['roi_bottom']}, pixels {stats['min']}-"
                  f"{stats['max']} (mean {stats['mean']:.0f}), suggested "
                  f"threshold ~{stats['otsu']:.0f}")
        colors = [c for _, c in result["boundary_xs"]]
        n = len(colors)
        n_yellow = colors.count("yellow")
        n_black = colors.count("black")
        self._log(f"Calibration: {n} boundary line(s) detected in this "
                  f"frame ({n_black} black, {n_yellow} yellow)."
                  + ("" if n >= 2 else " Aim the camera so at least two "
                                       "lines are inside the ROI band."))
        self.after(0, self.black_thresh_var.set, int(stats["otsu"]))
        self._on_camera_frame(result["debug_frame"], {})

    # ── Autonomous toggling / confirmation ──────────────────────────────────
    def _toggle_autonomous(self):
        if self.ctrl.autonomous:
            self.ctrl.stop_autonomous()
            self.mission.lane_following_stopped_by_user()
            self.auto_btn.config(text="START AUTO (L)", bg=self.GREEN)
            self.confirm_frame.pack_forget()
            return
        try:
            cam_idx = int(float(self.cam_index_var.get()))
        except ValueError:
            messagebox.showerror("Invalid", "Camera index must be an integer.")
            return
        road_type = self.road_type_var.get()
        if road_type not in (LaneKeepController.ROAD_TYPE_2_BLACK,
                             LaneKeepController.ROAD_TYPE_YELLOW):
            messagebox.showinfo("Road type required",
                                "Please select a STREET TYPE (2 black "
                                "borders or yellow follower) before "
                                "starting lane tracking.")
            return
        self._log(f"Road type selected: {road_type}")
        ok = self.ctrl.start_autonomous(
            camera_index=cam_idx,
            frame_cb=self._on_camera_frame,
            confirm_cb=self._on_confirm_request,
            params_cb=self._get_lk_params,
            road_type=road_type)
        if ok:
            self.auto_btn.config(text="STOP AUTO (L)", bg=self.RED,
                                 activebackground="#cc3333")
            self._last_auto_camera_index = cam_idx
            self._last_auto_road_type = road_type
            self.mission.begin_lane_following()

    def _restart_lane_follow(self):
        self.after(0, self._restart_lane_follow_gui)

    def _restart_lane_follow_gui(self):
        if self.ctrl.autonomous:
            return
        road_type = self._last_auto_road_type or self.road_type_var.get()
        ok = self.ctrl.start_autonomous(
            camera_index=self._last_auto_camera_index,
            frame_cb=self._on_camera_frame,
            confirm_cb=self._on_confirm_request,
            params_cb=self._get_lk_params,
            road_type=road_type)
        if ok:
            self.auto_btn.config(text="STOP AUTO (L)", bg=self.RED,
                                 activebackground="#cc3333")
        else:
            self._log("[MISSION] Could not auto-restart lane following - "
                      "press START AUTO manually.")

    def _on_confirm_request(self, topo):
        self.after(0, self._show_confirm_panel, topo)

    def _show_confirm_panel(self, topo):
        names = topo.get("lane_names", [])
        pos = topo.get("position", "?")
        lane_type = topo.get("lane_type", "unknown")
        type_txt = {"black_yellow": "BLACK_YELLOW - the vehicle will drive "
                                    "by FOLLOWING THE YELLOW LINE directly",
                   "2_blacks": "2_BLACKS (drive centered between both "
                               "black borders)"
                   }.get(lane_type, lane_type)
        self.confirm_lbl.config(
            text=f"Detected {len(names)} lane(s): {', '.join(names)}.\n"
                 f"Current lane looks like: {pos}.\n"
                 f"Lane type: {type_txt}.\n"
                 "Confirming will also CALIBRATE the system type + "
                 "first-image geometry, and SEED the home-lane width "
                 "memory used to stay anchored to this exact lane.\n"
                 "Is this correct, or should the vehicle track a "
                 "neighbouring lane instead?")
        self.confirm_frame.pack(fill="x", pady=(6, 0))

    def _confirm_lane(self, shift):
        self.confirm_frame.pack_forget()
        self.ctrl.confirm_lane(shift)

    # ── Frames from the worker thread ─────────────────────────────────────────
    def _on_camera_frame(self, frame_bgr, info):
        with self._frame_lock:
            self._latest_frame = (frame_bgr, info)
        self.mission.on_lane_frame(frame_bgr)

    def _refresh_camera_preview(self):
        if PIL_AVAILABLE:
            with self._frame_lock:
                payload = self._latest_frame
                self._latest_frame = None
            if payload is not None:
                frame, info = payload
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(rgb).resize((300, 225))
                    self._camera_photo = ImageTk.PhotoImage(img)
                    self.camera_canvas.delete("all")
                    self.camera_canvas.create_image(0, 0, anchor="nw",
                                                    image=self._camera_photo)
                except Exception:
                    pass
                topo = info.get("topo") if info else None
                yf = info.get("yf") if info else None
                if topo:
                    home_w = topo.get("home_lane_width")
                    home_txt = f"{home_w:.0f}px" if home_w else "-"
                    self.lane_status_lbl.config(
                        text=f"state:{topo['state']}  pos:{topo['position']}  "
                             f"type:{topo.get('lane_type','-')}  "
                             f"home_w:{home_txt}  lanes:{len(topo['lanes'])}  "
                             f"err:{info.get('error', 0):+.2f}")
                elif yf:
                    lo, hi = yf["region"]
                    xtxt = "-" if yf["x"] is None else f"{yf['x']:.0f}"
                    self.lane_status_lbl.config(
                        text=f"state:{yf['state']}  region:[{lo:.0f},"
                             f"{hi:.0f}]  x:{xtxt}  "
                             f"err:{info.get('error', 0):+.2f}")
        self.after(66, self._refresh_camera_preview)

    # ── STATE 2 / STATE 3 mission panel (8080 stream + debug + status) ──────
    def _refresh_mission_panel(self):
        m = self.mission

        status_txt = {
            MissionState.IDLE:           "IDLE - press START AUTO on the STATE 1 tab",
            MissionState.LANE_FOLLOWING: "STATE 1 - LANE FOLLOWING (watching for AprilTag / stop sign)",
            MissionState.TAG_TRANSITION: "TAG CONFIRMED - transitioning to centering...",
            MissionState.CENTERING:      "STATE 2 - CENTERING ON APRILTAG (PID active)",
            MissionState.STOPSIGN_HALT:  "STATE 2 - STOP SIGN HALT",
            MissionState.GRIPPER:        "STATE 3 - GRIPPER ACTIVE",
            MissionState.AWAITING_RESET: "CYCLE COMPLETE - press RESET & REPEAT",
            MissionState.ERROR:          "MISSION ERROR - see log",
        }.get(m.state, m.state.name)
        if m.is_estopped():
            status_txt = "⏹ E-STOP - mission ticking paused"
        if m.state is MissionState.GRIPPER and m.gripper_phase is not None:
            status_txt += f"  ({m.gripper_phase.name})"
        color = self.RED if (m.is_estopped() or m.state is MissionState.ERROR) \
            else (self.GREEN if m.state is MissionState.AWAITING_RESET else self.YELLOW)
        self.mission_status_lbl.config(text=status_txt, fg=color)

        self.mission_pose_lbl.config(
            text=(f"Distance  : {m.distance_cm:.1f} cm\n"
                  f"Left/Right: {m.offset_cm:.1f} cm\n"
                  f"Yaw       : {m.yaw_deg:.1f} deg\n"
                  f"Margin    : {m.last_margin:.1f}"))
        self.mission_center_lbl.config(
            text=(f"Pixel err : {m.pixel_error:.1f} px\n"
                  f"P/I/D     : {m.pid.last_p:.1f}/{m.pid.last_i:.1f}/{m.pid.last_d:.1f}\n"
                  f"Output mm : {m.pid_output_mm:.1f}\n"
                  f"Cmd deg   : {m.cmd_degrees}"))
        enc_txt = "-" if m.gripper_last_enc is None else str(m.gripper_last_enc)
        phase_txt = m.gripper_phase.name if m.gripper_phase else "-"
        self.mission_grip_lbl.config(
            text=(f"Phase : {phase_txt}\n"
                  f"SET3  : {m.gripper_angle}\n"
                  f"ENC3  : {enc_txt} (stable {m.gripper_stable_count}/{GRIPPER_HOMING_STABLE_COUNT})"))

        if m.state is MissionState.AWAITING_RESET:
            self.mission_reset_btn.config(bg=self.GREEN)
        else:
            self.mission_reset_btn.config(bg="#2a3040")

        if PIL_AVAILABLE:
            try:
                frame = m.frame_q.get_nowait()
            except queue.Empty:
                frame = None
            if frame is not None:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cw = max(self.mission_canvas.winfo_width(), 480)
                    ch = max(self.mission_canvas.winfo_height(), 360)
                    fh, fw = rgb.shape[:2]
                    scale = min(cw / fw, ch / fh)
                    nw, nh = max(int(fw * scale), 1), max(int(fh * scale), 1)
                    rgb = cv2.resize(rgb, (nw, nh))
                    img = Image.fromarray(rgb)
                    self._mission_photo = ImageTk.PhotoImage(img)
                    self.mission_canvas.delete("all")
                    self.mission_canvas.create_image(cw // 2, ch // 2, anchor="center",
                                                     image=self._mission_photo)
                except Exception:
                    pass

        self.after(66, self._refresh_mission_panel)

    def _on_mission_reconnect(self):
        ip = self.mission_ip_var.get().strip()
        try:
            port = int(self.mission_port_var.get().strip())
        except ValueError:
            self._log("[MISSION] Invalid stream port")
            return
        self.mission.reader.reconnect(ip=ip, port=port)

    def _on_mission_apply_resolution(self):
        ip = self.mission_ip_var.get().strip()
        try:
            port = int(self.mission_port_var.get().strip())
        except ValueError:
            self._log("[MISSION] Invalid stream port")
            return
        threading.Thread(target=set_camera_resolution,
                         args=(ip, port, self.mission_res_var.get(), self._log),
                         daemon=True).start()

    def _on_mission_apply_pid(self):
        try:
            kp = float(self.tag_kp_var.get().strip())
            ki = float(self.tag_ki_var.get().strip())
            kd = float(self.tag_kd_var.get().strip())
            margin = float(self.tag_margin_var.get().strip())
        except ValueError:
            self._log("[MISSION] Invalid PID Tuning value(s) - Kp/Ki/Kd/Error mm must all be numbers")
            return
        if margin < 0:
            self._log("[MISSION] Error margin must be >= 0 - not applied")
            return
        self.mission.update_centering_params(kp=kp, ki=ki, kd=kd, error_margin_mm=margin)

    def _on_mission_apply_gripper(self):
        try:
            step = float(self.grip_step_var.get().strip())
            step_time = float(self.grip_step_time_var.get().strip())
            pause = float(self.grip_pause_var.get().strip())
        except ValueError:
            self._log("[MISSION] Invalid gripper tuning value(s) - step/step "
                      "time/pause must all be numbers")
            return
        if step <= 0 or step_time <= 0 or pause < 0:
            self._log("[MISSION] Step size/time must be > 0 and pause must be >= 0 - not applied")
            return
        self.mission.update_gripper_params(step_deg=step, step_time_sec=step_time,
                                           pause_sec=pause)

    def _on_mission_estop(self):
        self.mission.emergency_stop()

    def _on_mission_align(self):
        self.mission._abort_running_gripper_thread()
        self.ctrl.stop_autonomous()
        self.mission.lane_following_stopped_by_user()
        self.ctrl.set_mission_active(True)
        self.ctrl.send_stop_command()
        self.mission.trigger = "apriltag"
        self.mission._set_state(MissionState.TAG_TRANSITION)
        self._log("[MISSION] Manual override -> ALIGN (State 2) requested")

    def _on_mission_grip(self):
        self.ctrl.stop_autonomous()
        self.mission.lane_following_stopped_by_user()
        self.ctrl.set_mission_active(True)
        self._log("[MISSION] Manual override -> GRIPPER (State 3) requested")
        self.mission._enter_gripper()

    def _on_mission_reset_repeat(self):
        self.mission.manual_reset_and_repeat()

    # ── Status callback ───────────────────────────────────────────────────────
    def _on_status(self, speed, steer):
        self.after(0, self._update_ui, speed, steer)

    def _update_ui(self, speed, steer):
        self.speed_lbl.config(text=f"{speed:+5d}")
        self.steer_lbl.config(text=f"{steer:5d}")
        self._update_bars(speed, steer)
        lo, hi = self.ctrl.steer_min, self.ctrl.steer_max
        norm_x = (steer - (lo + hi) / 2) / ((hi - lo) / 2)
        norm_x = clamp(-norm_x, -1, 1)
        norm_y = speed / SPEED_MAX_VALUE
        self._draw_joystick(norm_x, norm_y)

    # ── Connection ────────────────────────────────────────────────────────────
    def _toggle_connect(self):
        if self.tcp.connected:
            self.ctrl.stop_autonomous()
            self.mission._abort_running_gripper_thread(join_timeout=1.0)
            self.mission.trigger = None
            self.mission._set_state(MissionState.IDLE)
            self.auto_btn.config(text="START AUTO (L)", bg=self.GREEN)
            self.confirm_frame.pack_forget()
            self.tcp.disconnect()
            self.ctrl.stop()
            self.conn_btn.config(text="CONNECT", bg=self.GREEN)
            self._draw_indicator(False)
            self._log("Disconnected.")
            self.ctrl = VehicleController(self.tcp, self._on_status, self._log)
            self.mission.vc = self.ctrl
            self.ctrl.start()
        else:
            host = self.host_var.get().strip()
            if not host:
                messagebox.showerror("Vehicle IP required",
                                     "Enter the IP address of the vehicle "
                                     "controller (TCP port 4545).")
                return
            try:
                port = int(self.port_var.get().strip())
            except ValueError:
                messagebox.showerror("Error", "Port must be an integer.")
                return
            result = self.tcp.connect(host, port)
            if result is True:
                self.conn_btn.config(text="DISCONNECT", bg=self.RED,
                                     activebackground="#cc3333")
                self._draw_indicator(True)
                self._log(f"Connected to {host}:{port}")
            else:
                messagebox.showerror("Connection Failed", str(result))

    def _draw_indicator(self, on):
        c = self.conn_indicator
        c.delete("all")
        c.create_oval(1, 1, 11, 11, fill=self.GREEN if on else self.RED,
                      outline="")

    # ── Servo apply ────────────────────────────────────────────────────────────
    def _apply_servo(self):
        try:
            center = int(self.center_var.get())
            lo = int(self.min_var.get())
            hi = int(self.max_var.get())
        except ValueError:
            messagebox.showerror("Invalid", "Servo values must be integers.")
            return
        if not (lo <= center <= hi):
            messagebox.showerror("Invalid", "Must satisfy: Min <= Center <= Max.")
            return
        self.ctrl.steer_center = center
        self.ctrl.steer_min = lo
        self.ctrl.steer_max = hi
        self.ctrl.steer = center
        self.ctrl.send_servo_config()

    # ── Log ─────────────────────────────────────────────────────────────────────
    def _log(self, msg):
        self.after(0, self._append_log, msg)

    def _append_log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_box.config(state="normal")
        self.log_box.insert("end", f"[{ts}] {msg}\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    # ── Cleanup ──────────────────────────────────────────────────────────────────
    def _on_close(self):
        self.mission.halt()
        self.ctrl.stop()
        self.tcp.disconnect()
        self.destroy()
