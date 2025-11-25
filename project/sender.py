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
from sprite import StickmanSprite, SPRITE_WIDTH, SPRITE_HEIGHT


HOST = "0.0.0.0"
PORT = 5000          # video stream port
CONTROL_PORT = 5001  # control port

MOVE_SPEED = 8
GRAVITY = 1.0
JUMP_VELOCITY = -16.0

INPUT_HOLD_MS = 80   # how long movement intent persists after a packet
PUNCH_MS = 220       # how long punch animation stays active

GROUND_Y = None  # set after we know screen height

square_state = {
    "x": 100.0,
    "y": 100.0,
    "vy": 0.0,
    "on_ground": False,
    "on_wall": False,
    "facing": 1,           # 1=right, -1=left
    "input_vx": 0.0,       # horizontal intent
    "input_ttl_ms": 0,     # how long intent remains active
    "punch_ttl_ms": 0,     # countdown for punch animation
}


def recvall(sock, n: int):
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


def left_click():
    """Send one OS left-click (Windows only)."""
    if sys.platform != "win32":
        return
    user32 = ctypes.windll.user32
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def screen_sender():
    """
    Stream the main monitor as JPEG frames to a receiver.
    Protocol:
      - send 4 bytes: frame length (big endian)
      - send frame bytes
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)

        print(f"[sender] Waiting for video connection on {HOST}:{PORT}...")
        conn, addr = s.accept()
        print(f"[sender] Video connection from {addr}")

        with conn:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                while True:
                    img = np.array(sct.grab(monitor))
                    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                    if not ok:
                        continue

                    payload = encoded.tobytes()
                    header = struct.pack("!I", len(payload))

                    try:
                        conn.sendall(header)
                        conn.sendall(payload)
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        print("[sender] Video connection closed")
                        break


def control_server():
    """
    Get movement commands from the receiver and update control intent.
    Message format: 12 bytes, !iii (dx, dy, action).
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
                    data = recvall(conn, 12)
                    if not data:
                        break

                    dx, dy, action = struct.unpack("!iii", data)

                    if action == 1 and square_state.get("punch_ttl_ms", 0) == 0:
                        square_state["punch_ttl_ms"] = PUNCH_MS
                        left_click()

                    # Horizontal: treat any nonzero dx as "keep moving this direction briefly".
                    # This bridges OpenCV's key-repeat delay and allows moving while jumping.
                    if dx != 0:
                        square_state["input_vx"] = MOVE_SPEED if dx > 0 else -MOVE_SPEED
                        square_state["input_ttl_ms"] = INPUT_HOLD_MS
                        square_state["facing"] = 1 if dx > 0 else -1

                    # Jump if on ground or on a wall
                    if dy < 0 and (square_state.get("on_ground", False) or square_state.get("on_wall", False)):
                        square_state["vy"] = JUMP_VELOCITY
                        square_state["on_ground"] = False
                        square_state["on_wall"] = False

            except (ConnectionResetError, OSError):
                print("[sender] Control connection error")
            finally:
                print("[sender] Control server thread exiting")


class Overlay(QtWidgets.QWidget):
    """Transparent overlay that draws the sprite and window blocks."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("PythonOverlay")
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
            | QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)

        screen = QtWidgets.QApplication.primaryScreen()
        geo = screen.geometry()
        self.screen_w = geo.width()
        self.screen_h = geo.height()

        global GROUND_Y
        GROUND_Y = self.screen_h - SPRITE_HEIGHT - 10

        self.platforms = []
        self.sprite = StickmanSprite(width=SPRITE_WIDTH, height=SPRITE_HEIGHT)

        self.prev_time = time.time()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)

    def tick(self):
        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now
        dt_ms = int(dt * 1000)

        # refresh platforms from windows
        self.platforms = get_window_platforms()

        # decay intent + punch timer
        if square_state["input_ttl_ms"] > 0:
            square_state["input_ttl_ms"] = max(0, square_state["input_ttl_ms"] - dt_ms)
        else:
            square_state["input_vx"] = 0.0

        if square_state.get("punch_ttl_ms", 0) > 0:
            square_state["punch_ttl_ms"] = max(0, int(square_state["punch_ttl_ms"]) - dt_ms)

        # --- horizontal physics ---
        prev_x = square_state["x"]
        square_state["x"] += float(square_state["input_vx"])

        # keep on screen
        square_state["x"] = max(0, min(self.screen_w - SPRITE_WIDTH, square_state["x"]))

        # --- vertical physics ---
        prev_y = square_state["y"]
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
            square_state["vy"] = 0
            square_state["on_ground"] = True

        self.update()

    def handle_vertical_collisions(self, prev_y):
        sx = square_state["x"]
        sy = square_state["y"]
        vy = square_state["vy"]

        # only resolve downward collisions as "landing"
        if vy < 0:
            return

        sprite_bottom = sy + SPRITE_HEIGHT
        prev_bottom = prev_y + SPRITE_HEIGHT

        for (px, py, pw, ph) in self.platforms:
            platform_top = py
            sprite_left = sx
            sprite_right = sx + SPRITE_WIDTH

            if sprite_right <= px or sprite_left >= px + pw:
                continue

            if prev_bottom <= platform_top <= sprite_bottom:
                # land on top
                square_state["y"] = platform_top - SPRITE_HEIGHT
                square_state["vy"] = 0
                square_state["on_ground"] = True
                return

    def handle_horizontal_collisions(self, prev_x):
        sx = square_state["x"]
        sy = square_state["y"]

        sprite_top = sy
        sprite_bottom = sy + SPRITE_HEIGHT

        for (px, py, pw, ph) in self.platforms:
            platform_left = px
            platform_right = px + pw
            platform_top = py
            platform_bottom = py + ph

            if sprite_bottom <= platform_top or sprite_top >= platform_bottom:
                continue

            # moving right into left side
            if sx > prev_x:
                prev_right = prev_x + SPRITE_WIDTH
                curr_right = sx + SPRITE_WIDTH
                if prev_right <= platform_left <= curr_right:
                    square_state["x"] = platform_left - SPRITE_WIDTH
                    square_state["on_wall"] = True
                    square_state["input_ttl_ms"] = 0
                    return

            # moving left into right side
            if sx < prev_x:
                prev_left = prev_x
                curr_left = sx
                if curr_left <= platform_right <= prev_left:
                    square_state["x"] = platform_right
                    square_state["on_wall"] = True
                    square_state["input_ttl_ms"] = 0
                    return

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # thin green line to show window tops
        platform_color = QtGui.QColor(0, 255, 0, 80)
        for (px, py, pw, ph) in self.platforms:
            rect = QtCore.QRect(int(px), int(py), int(pw), 5)
            painter.fillRect(rect, platform_color)

        # draw sprite
        moving = abs(square_state.get("input_vx", 0.0)) > 0.1
        self.sprite.draw(
            painter,
            int(square_state["x"]),
            int(square_state["y"]),
            facing=square_state.get("facing", 1),
            moving=moving,
            on_ground=bool(square_state.get("on_ground", False)),
            vy=float(square_state.get("vy", 0.0)),
            punching=bool(square_state.get("punch_ttl_ms", 0) > 0),
        )


def start_overlay():
    app = QtWidgets.QApplication(sys.argv)
    overlay = Overlay()

    screen = QtWidgets.QApplication.primaryScreen()
    geo = screen.geometry()
    overlay.setGeometry(geo)

    overlay.show()

    # Make it click-through on Windows (keeps NO mouse capture)
    if sys.platform == "win32":
        hwnd = int(overlay.winId())

        user32 = ctypes.windll.user32
        GWL_EXSTYLE = -20
        WS_EX_LAYERED = 0x80000
        WS_EX_TRANSPARENT = 0x20

        old_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(
            hwnd,
            GWL_EXSTYLE,
            old_style | WS_EX_LAYERED | WS_EX_TRANSPARENT
        )

    print("[sender] Overlay started (gravity, window terrain, wall jumps)")
    app.exec_()


if __name__ == "__main__":
    threading.Thread(target=screen_sender, daemon=True).start()
    threading.Thread(target=control_server, daemon=True).start()
    start_overlay()
