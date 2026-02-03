"""
Air Canvas  –  Full GUI Desktop App  (fixed)
=============================================
Requirements:  pip install opencv-python numpy Pillow
Run:           python air_canvas.py
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw
import threading
import time
import os

# ─────────────────────────────────────────────
# COLOUR PALETTE   HSV lo/hi  +  RGB draw colour
# ─────────────────────────────────────────────
COLOURS = {
    "Blue":   {"hsv_lo": (100, 150,  50), "hsv_hi": (130, 255, 255), "rgb": (30,  120, 255)},
    "Green":  {"hsv_lo": ( 40, 100,  50), "hsv_hi": ( 75, 255, 255), "rgb": (46,  204,  71)},
    "Red":    {"hsv_lo": (  0, 150,  50), "hsv_hi": ( 12, 255, 255), "rgb": (231,  76,  60),
               "hsv_lo2":(168, 150,  50), "hsv_hi2":(180, 255, 255)},
    "Yellow": {"hsv_lo": ( 20, 100, 100), "hsv_hi": ( 35, 255, 255), "rgb": (241, 196,  15)},
    "Purple": {"hsv_lo": (130, 100,  50), "hsv_hi": (160, 255, 255), "rgb": (155,  89, 182)},
    "Orange": {"hsv_lo": ( 12, 150,  80), "hsv_hi": ( 20, 255, 255), "rgb": (230, 126,  34)},
}

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
CAM_W, CAM_H       = 960, 540
CANVAS_W, CANVAS_H = 960, 540
MIN_AREA            = 400
MORPH_K             = 7


# ─────────────────────────────────────────────
# COLOUR DETECTION  (pure, no side-effects)
# ─────────────────────────────────────────────
def _make_mask(hsv, name):
    c = COLOURS[name]
    m = cv2.inRange(hsv, np.array(c["hsv_lo"]), np.array(c["hsv_hi"]))
    if "hsv_lo2" in c:
        m |= cv2.inRange(hsv, np.array(c["hsv_lo2"]), np.array(c["hsv_hi2"]))
    return m


def _detect_tip(mask):
    cts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cts:
        return None
    big = max(cts, key=cv2.contourArea)
    if cv2.contourArea(big) < MIN_AREA:
        return None
    M = cv2.moments(big)
    if M["m00"] == 0:
        return None
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


def _open_camera():
    """Try indices 0-3; return the first that opens, or None."""
    for idx in range(4):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
            return cap
        cap.release()
    return None


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
class AirCanvas:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Air Canvas")
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(False, False)

        # ── shared state ─────────────────────
        self.running    = True
        self.drawing    = True
        self.show_mask  = False
        self.sel_colour = "Blue"
        self.brush_size = 4

        # pen tracking  –  only touched on main thread
        self.prev_point  = None
        self.pen_is_down = False   # True once we see the first tip in a stroke

        # canvas  –  only mutated on the main thread
        self.canvas_img  = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
        self.canvas_draw = ImageDraw.Draw(self.canvas_img)
        self.undo_stack  = []      # max 30 snapshots

        # ── thread communication  –  lock-protected ──
        self._lock         = threading.Lock()
        self._latest_frame = None   # BGR ndarray
        self._latest_mask  = None   # uint8 ndarray
        self._latest_tip   = None   # (x, y) or None

        # ── camera ───────────────────────────
        self.cap = _open_camera()
        if self.cap is None:
            messagebox.showerror("Camera Error",
                                 "Could not open any webcam.\n"
                                 "Make sure a camera is connected and not in use.")
            root.destroy()
            return

        # ── build the window ─────────────────
        self._build_ui()

        # ── start background camera reader ───
        self._cam_thread = threading.Thread(target=self._cam_loop, daemon=True)
        self._cam_thread.start()

        # ── kick off the Tk update cycle ─────
        self.root.after(16, self._update_loop)

        # close hook
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ────────────────────────────────────────────────────
    # UI  LAYOUT
    # ────────────────────────────────────────────────────
    def _build_ui(self):
        # ── top toolbar ──────────────────────
        top = tk.Frame(self.root, bg="#16213e", pady=8, padx=12)
        top.pack(fill="x")

        tk.Label(top, text="AIR CANVAS", font=("Courier New", 18, "bold"),
                 bg="#16213e", fg="#e94560").pack(side="left", padx=(0, 24))

        # colour row
        cf = tk.Frame(top, bg="#16213e")
        cf.pack(side="left")
        tk.Label(cf, text="COLOUR", font=("Courier New", 8),
                 bg="#16213e", fg="#a0a0b0").pack(anchor="w")
        btn_row = tk.Frame(cf, bg="#16213e")
        btn_row.pack()

        self._col_btns = {}
        for name, info in COLOURS.items():
            r, g, b   = info["rgb"]
            hex_col   = f"#{r:02x}{g:02x}{b:02x}"
            btn = tk.Button(btn_row, width=3, height=1,
                            bg=hex_col, activebackground=hex_col,
                            relief="groove", bd=2, cursor="hand2",
                            command=lambda n=name: self._pick_colour(n))
            btn.pack(side="left", padx=2)
            self._col_btns[name] = btn
        self._pick_colour("Blue")

        # brush size slider
        bf = tk.Frame(top, bg="#16213e")
        bf.pack(side="left", padx=(20, 0))
        tk.Label(bf, text="BRUSH SIZE", font=("Courier New", 8),
                 bg="#16213e", fg="#a0a0b0").pack(anchor="w")
        self._brush_var = tk.IntVar(value=self.brush_size)
        tk.Scale(bf, from_=1, to=30, orient="horizontal", length=130,
                 variable=self._brush_var, bg="#16213e", fg="#e94560",
                 troughcolor="#0f3460", highlightthickness=0,
                 command=lambda _: self._on_brush_change()
                 ).pack()

        # action buttons – right-aligned
        af = tk.Frame(top, bg="#16213e")
        af.pack(side="right")
        self._btn(af, "UNDO",  self._undo).pack(side="left", padx=2)
        self._btn(af, "CLEAR", self._clear).pack(side="left", padx=2)
        self._btn(af, "SAVE",  self._save).pack(side="left", padx=2)

        # ── toggle row ───────────────────────
        mid = tk.Frame(self.root, bg="#1a1a2e", pady=3)
        mid.pack(fill="x")

        self._draw_var = tk.BooleanVar(value=True)
        self._mask_var = tk.BooleanVar(value=False)

        tk.Checkbutton(mid, text="  Drawing ON", variable=self._draw_var,
                       font=("Courier New", 9), bg="#1a1a2e", fg="#ccc",
                       activebackground="#1a1a2e", activeforeground="#ccc",
                       selectcolor="#0f3460",
                       command=self._toggle_draw
                       ).pack(side="left", padx=12)

        tk.Checkbutton(mid, text="  Show Mask", variable=self._mask_var,
                       font=("Courier New", 9), bg="#1a1a2e", fg="#ccc",
                       activebackground="#1a1a2e", activeforeground="#ccc",
                       selectcolor="#0f3460",
                       command=self._toggle_mask
                       ).pack(side="left", padx=6)

        # live status (right)
        self._status = tk.StringVar(value="Starting…")
        tk.Label(mid, textvariable=self._status, font=("Courier New", 9),
                 bg="#1a1a2e", fg="#6c757d").pack(side="right", padx=12)

        # ── video label ──────────────────────
        vid_frame = tk.Frame(self.root, bg="#0f3460", bd=3)
        vid_frame.pack(pady=(4, 4))
        self._vid_label = tk.Label(vid_frame, bg="#0f3460",
                                   width=CANVAS_W, height=CANVAS_H)
        self._vid_label.pack()

        # ── hint bar ─────────────────────────
        hint = tk.Frame(self.root, bg="#16213e", pady=4)
        hint.pack(fill="x")
        tk.Label(hint,
                 text="Hold a coloured marker in front of your webcam and move it to draw.",
                 font=("Courier New", 9), bg="#16213e", fg="#6c757d"
                 ).pack()

    # ────────────────────────────────────────────────────
    # TOOLBAR  HELPERS
    # ────────────────────────────────────────────────────
    @staticmethod
    def _btn(parent, text, cmd):
        return tk.Button(parent, text=text, command=cmd,
                         font=("Courier New", 9, "bold"),
                         bg="#0f3460", fg="#e94560",
                         activebackground="#e94560", activeforeground="#fff",
                         relief="flat", bd=0, padx=12, pady=4, cursor="hand2")

    def _pick_colour(self, name):
        self.sel_colour = name
        self.prev_point = None
        self.pen_is_down = False
        for n, btn in self._col_btns.items():
            if n == name:
                btn.config(relief="sunken", bd=3)
            else:
                btn.config(relief="groove", bd=2)

    def _on_brush_change(self):
        self.brush_size = self._brush_var.get()

    def _toggle_draw(self):
        self.drawing = self._draw_var.get()
        if not self.drawing:
            self.prev_point  = None
            self.pen_is_down = False

    def _toggle_mask(self):
        self.show_mask = self._mask_var.get()

    # ────────────────────────────────────────────────────
    # CANVAS  OPERATIONS  (main thread only)
    # ────────────────────────────────────────────────────
    def _snapshot(self):
        self.undo_stack.append(self.canvas_img.copy())
        if len(self.undo_stack) > 30:
            self.undo_stack.pop(0)

    def _undo(self):
        if self.undo_stack:
            self.canvas_img  = self.undo_stack.pop()
            self.canvas_draw = ImageDraw.Draw(self.canvas_img)
            self.prev_point  = None
            self.pen_is_down = False

    def _clear(self):
        self._snapshot()
        self.canvas_img  = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
        self.canvas_draw = ImageDraw.Draw(self.canvas_img)
        self.prev_point  = None
        self.pen_is_down = False

    def _save(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
            initialfile="air_canvas.png",
            title="Save Drawing")
        if path:
            bg = Image.new("RGB", (CANVAS_W, CANVAS_H), (255, 255, 255))
            bg.paste(self.canvas_img, mask=self.canvas_img.split()[3])
            bg.save(path)
            self._status.set("Saved: " + os.path.basename(path))

    # ────────────────────────────────────────────────────
    # CAMERA  THREAD  –  reads frames + detects tip only
    # ────────────────────────────────────────────────────
    def _cam_loop(self):
        kernel = np.ones((MORPH_K, MORPH_K), np.uint8)
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (CAM_W, CAM_H))

            hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = _make_mask(hsv, self.sel_colour)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
            tip  = _detect_tip(mask)

            # draw indicator circle directly on the BGR frame
            if tip is not None:
                cv2.circle(frame, tip, self.brush_size + 6, (255, 255, 255), 2)
                cv2.circle(frame, tip, self.brush_size + 2,
                           tuple(int(v) for v in reversed(COLOURS[self.sel_colour]["rgb"])), -1)

            # hand off to main thread
            with self._lock:
                self._latest_frame = frame
                self._latest_mask  = mask
                self._latest_tip   = tip

            time.sleep(0.01)   # ~100 fps

    # ────────────────────────────────────────────────────
    # TK  UPDATE  LOOP   (~60 fps)
    # runs entirely on the main thread
    # ────────────────────────────────────────────────────
    def _update_loop(self):
        if not self.running:
            return

        # grab latest data from camera thread
        with self._lock:
            frame = self._latest_frame
            mask  = self._latest_mask
            tip   = self._latest_tip

        if frame is not None:
            # ── draw a line segment if pen is moving ──
            if self.drawing and tip is not None:
                if not self.pen_is_down:
                    # pen just touched down  –  snapshot before we start this stroke
                    self._snapshot()
                    self.pen_is_down = True
                    self.prev_point  = tip          # first point, no line yet
                else:
                    # continuing stroke
                    if self.prev_point is not None:
                        r, g, b = COLOURS[self.sel_colour]["rgb"]
                        self.canvas_draw.line(
                            [self.prev_point, tip],
                            fill=(r, g, b, 255),
                            width=self.brush_size)
                    self.prev_point = tip
            else:
                # marker not visible  –  pen lifted
                self.prev_point  = None
                self.pen_is_down = False

            # ── composite for display ──────────────────
            if self.show_mask and mask is not None:
                display_base = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            else:
                display_base = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pil_base = Image.fromarray(display_base)

            # overlay the drawing (RGBA  →  paste with alpha mask)
            pil_base.paste(self.canvas_img, mask=self.canvas_img.split()[3])

            # push to label
            self._tk_img = ImageTk.PhotoImage(pil_base)
            self._vid_label.config(image=self._tk_img)

            # update status text
            state = "Drawing" if self.drawing else "Paused"
            self._status.set(f"{state}  |  Colour: {self.sel_colour}  |  "
                             f"Brush: {self.brush_size}")

        # schedule next tick
        self.root.after(16, self._update_loop)

    # ────────────────────────────────────────────────────
    # CLEAN SHUTDOWN
    # ────────────────────────────────────────────────────
    def _on_close(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = AirCanvas(root)
    root.mainloop()