import socket
import struct
import time
import threading
import sys
import ctypes

import cv2
import mss
import numpy as np

from PyQt5 import QtWidgets, QtCore, QtGui

from window_platforms import get_window_platforms

HOST = "0.0.0.0"
PORT = 5000          # video stream port
CONTROL_PORT = 5001  # control port

# square physics
SQUARE_SIZE = 100
GRAVITY = 1.0          # pixels per frame^2
JUMP_VELOCITY = -20.0  # jump speed

square_state = {
    "x": 100.0,
    "y": 100.0,
    "vy": 0.0,
    "on_ground": False,
    "on_wall": False,
}

GROUND_Y = None  # set after we know screen height


def recvall(sock, n):
    """Receive exactly n bytes or return None."""
    data = bytearray()
    while len(data) < n:
        try:
            packet = sock.recv(n - len(data))
        except OSError:
            return None
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


def screen_sender():
    """Send desktop frames to the receiver."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((HOST, PORT))
        server_sock.listen(1)

        print(f"[sender] Waiting for video connection on {HOST}:{PORT}...")
        conn, addr = server_sock.accept()
        print(f"[sender] Video connection from {addr}")

        with conn, mss.mss() as sct:
            monitor = sct.monitors[1]  # primary display
            print(f"[sender] Capturing display: {monitor['width']}x{monitor['height']}")

            try:
                while True:
                    sct_img = sct.grab(monitor)
                    frame = np.array(sct_img)

                    # remove alpha channel if present
                    if frame.shape[2] == 4:
                        frame = frame[:, :, :3]

                    ok, encoded = cv2.imencode(
                        ".jpg",
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 50],
                    )
                    if not ok:
                        continue

                    data = encoded.tobytes()
                    header = struct.pack("!I", len(data))
                    conn.sendall(header + data)

                    time.sleep(0.05)

            except (BrokenPipeError, ConnectionResetError):
                print("[sender] Video connection closed by receiver")
            finally:
                print("[sender] Screen sender thread exiting")


def control_server():
    """
    Get movement commands from the receiver and update the square.
    Message format: 8 bytes, !ii (dx, dy).
    """
    global square_state

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((HOST, CONTROL_PORT))
        server_sock.listen(1)

        print(f"[sender] Waiting for control connection on {HOST}:{CONTROL_PORT}...")
        conn, addr = server_sock.accept()
        print(f"[sender] Control connection from {addr}")

        with conn:
            try:
                while True:
                    data = recvall(conn, 8)
                    if data is None:
                        print("[sender] Control connection closed")
                        break

                    dx, dy = struct.unpack("!ii", data)

                    # horizontal move
                    square_state["x"] += float(dx)

                    # jump if on ground or on a wall
                    if dy < 0 and (square_state.get("on_ground", False) or
                                   square_state.get("on_wall", False)):
                        square_state["vy"] = JUMP_VELOCITY
                        square_state["on_ground"] = False
                        square_state["on_wall"] = False

            except (ConnectionResetError, OSError):
                print("[sender] Control connection error")
            finally:
                print("[sender] Control server thread exiting")


class Overlay(QtWidgets.QWidget):
    """Transparent overlay that draws the square and window blocks."""

    def __init__(self):
        super().__init__()

        # window style: no border, always on top, not in taskbar
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.Tool
        )
        self.setWindowTitle("PythonOverlay")

        # allow transparent background
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)

        # cover the main screen
        screen_geom = QtWidgets.QApplication.primaryScreen().geometry()
        self.setGeometry(screen_geom)

        # ground is the bottom of the screen
        global GROUND_Y, square_state
        GROUND_Y = self.height() - SQUARE_SIZE
        square_state["y"] = float(GROUND_Y)
        square_state["vy"] = 0.0
        square_state["on_ground"] = True

        # (x, y, w, h) rectangles for windows
        self.platforms = []

        # physics update timer
        self.physics_timer = QtCore.QTimer(self)
        self.physics_timer.timeout.connect(self.step_physics)
        self.physics_timer.start(16)

        # refresh window rectangles every second
        self.platform_timer = QtCore.QTimer(self)
        self.platform_timer.timeout.connect(self.refresh_platforms)
        self.platform_timer.start(1000)

    def refresh_platforms(self):
        """Reload window rectangles."""
        try:
            self.platforms = get_window_platforms()
        except Exception as e:
            print(f"[sender] Could not update platforms: {e}")

    def step_physics(self):
        """Apply gravity, handle collisions, then redraw."""
        global square_state, GROUND_Y

        if GROUND_Y is None:
            return

        prev_x = square_state["x"]
        prev_y = square_state["y"]

        # gravity
        square_state["vy"] += GRAVITY
        square_state["y"] += square_state["vy"]

        square_state["on_ground"] = False
        square_state["on_wall"] = False

        # collide with windows
        self.handle_vertical_collisions(prev_y)
        self.handle_horizontal_collisions(prev_x)

        # ground at bottom of screen
        if square_state["y"] >= GROUND_Y and square_state["vy"] >= 0:
            square_state["y"] = GROUND_Y
            square_state["vy"] = 0.0
            square_state["on_ground"] = True

        # keep inside screen horizontally
        if square_state["x"] < 0:
            square_state["x"] = 0.0
        elif square_state["x"] > self.width() - SQUARE_SIZE:
            square_state["x"] = float(self.width() - SQUARE_SIZE)

        self.update()

    def handle_vertical_collisions(self, prev_y):
        """Handle landing on top of windows and hitting their bottoms."""
        global square_state

        size = SQUARE_SIZE
        x = square_state["x"]
        y = square_state["y"]
        vy = square_state["vy"]

        if vy == 0:
            return

        top_prev = prev_y
        bottom_prev = prev_y + size
        top_cur = y
        bottom_cur = y + size

        for (px, py, pw, ph) in self.platforms:
            plat_left = px
            plat_right = px + pw
            plat_top = py
            plat_bottom = py + ph

            # must overlap horizontally
            if x + size <= plat_left or x >= plat_right:
                continue

            if vy > 0:
                # moving down: crossing top edge
                if bottom_prev <= plat_top <= bottom_cur:
                    square_state["y"] = plat_top - size
                    square_state["vy"] = 0.0
                    square_state["on_ground"] = True
                    y = square_state["y"]
                    top_cur = y
                    bottom_cur = y + size
            else:
                # moving up: crossing bottom edge
                if top_prev >= plat_bottom >= top_cur:
                    square_state["y"] = plat_bottom
                    if square_state["vy"] < 0:
                        square_state["vy"] = 0.0
                    y = square_state["y"]
                    top_cur = y
                    bottom_cur = y + size

    def handle_horizontal_collisions(self, prev_x):
        """Handle running into window sides (walls)."""
        global square_state

        size = SQUARE_SIZE
        x = square_state["x"]
        y = square_state["y"]

        dx = x - prev_x
        if dx == 0:
            return

        left_prev = prev_x
        right_prev = prev_x + size
        left_cur = x
        right_cur = x + size

        for (px, py, pw, ph) in self.platforms:
            plat_left = px
            plat_right = px + pw
            plat_top = py
            plat_bottom = py + ph

            # must overlap vertically
            if y + size <= plat_top or y >= plat_bottom:
                continue

            if dx > 0:
                # moving right, hit left edge
                if right_prev <= plat_left <= right_cur:
                    square_state["x"] = plat_left - size
                    square_state["on_wall"] = True
                    x = square_state["x"]
                    left_cur = x
                    right_cur = x + size
            else:
                # moving left, hit right edge
                if left_prev >= plat_right >= left_cur:
                    square_state["x"] = plat_right
                    square_state["on_wall"] = True
                    x = square_state["x"]
                    left_cur = x
                    right_cur = x + size

    def paintEvent(self, event):
        """Draw the windows (hint) and the red square."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # thin green line to show window tops
        platform_color = QtGui.QColor(0, 255, 0, 80)
        for (px, py, pw, ph) in self.platforms:
            rect = QtCore.QRect(int(px), int(py), int(pw), 5)
            painter.fillRect(rect, platform_color)

        # square
        x = int(square_state["x"])
        y = int(square_state["y"])
        square_rect = QtCore.QRect(x, y, SQUARE_SIZE, SQUARE_SIZE)

        square_color = QtGui.QColor(255, 0, 0, 255)
        painter.fillRect(square_rect, square_color)


def start_overlay():
    """Create the overlay window and make it click-through."""
    if sys.platform != "win32":
        print("[sender] Overlay only implemented on Windows; skipping overlay.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return
        return

    app = QtWidgets.QApplication(sys.argv)
    overlay = Overlay()
    overlay.showFullScreen()

    # make this window ignore mouse clicks
    hwnd = int(overlay.winId())

    GWL_EXSTYLE = -20
    WS_EX_LAYERED = 0x00080000
    WS_EX_TRANSPARENT = 0x00000020

    user32 = ctypes.windll.user32
    old_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongW(
        hwnd,
        GWL_EXSTYLE,
        old_style | WS_EX_LAYERED | WS_EX_TRANSPARENT
    )

    print("[sender] Overlay started (gravity, window terrain, wall jumps)")
    app.exec_()


if __name__ == "__main__":
    # start streaming and control threads
    threading.Thread(target=screen_sender, daemon=True).start()
    threading.Thread(target=control_server, daemon=True).start()

    # run the overlay GUI
    start_overlay()
