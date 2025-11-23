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

HOST = "0.0.0.0"
PORT = 5000          # video stream port
CONTROL_PORT = 5001  # control port for square movement

# --- square physics state ---
SQUARE_SIZE = 100
GRAVITY = 1.0          # pixels per frame^2 (tweak this)
JUMP_VELOCITY = -20.0  # initial upward velocity (tweak this)

square_state = {
    "x": 100.0,
    "y": 100.0,
    "vy": 0.0,
    "on_ground": False,
}

GROUND_Y = None  # set after we know the screen height


def recvall(sock, n):
    """Receive exactly n bytes (or None on EOF)."""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


def screen_sender():
    """Stream desktop frames to the receiver."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((HOST, PORT))
        server_sock.listen(1)

        print(f"[sender] Waiting for connection on {HOST}:{PORT}...")
        conn, addr = server_sock.accept()
        print(f"[sender] Connected by {addr}")

        with conn, mss.mss() as sct:
            monitor = sct.monitors[1]
            print(f"[sender] Capturing full display: "
                  f"{monitor['width']}x{monitor['height']}")

            try:
                while True:
                    sct_img = sct.grab(monitor)
                    frame = np.array(sct_img)

                    # Drop alpha channel if present
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
                print("[sender] Connection closed by receiver")
            finally:
                print("[sender] Shutting down screen sender")


def control_server():
    """
    Listen for control messages and update square_state.
    Protocol: 8 bytes per message: !ii (dx, dy).
    dx = left/right, any dy < 0 = jump.
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
                        print("[sender] Control connection closed by receiver")
                        break
                    dx, dy = struct.unpack("!ii", data)

                    # left/right
                    square_state["x"] += dx

                    # jump when w is pressed (dy < 0)
                    if dy < 0 and square_state.get("on_ground", False):
                        square_state["vy"] = JUMP_VELOCITY
                        square_state["on_ground"] = False

            except (ConnectionResetError, OSError):
                print("[sender] Control connection error, closing")


class Overlay(QtWidgets.QWidget):
    """Transparent overlay that draws the red square with gravity/jump."""

    def __init__(self):
        super().__init__()

        # Frameless, always on top, no taskbar entry
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.Tool
        )

        # Per-pixel transparency
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)

        # Full screen on primary monitor
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        self.setGeometry(screen)

        # Set ground level based on screen height
        global GROUND_Y, square_state
        GROUND_Y = self.height() - SQUARE_SIZE

        # Start the square on the ground
        square_state["y"] = float(GROUND_Y)
        square_state["vy"] = 0.0
        square_state["on_ground"] = True

        # Physics at ~60 FPS
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.step)
        self.timer.start(16)

    def step(self):
        """Update physics and redraw."""
        global square_state, GROUND_Y

        if GROUND_Y is None:
            return

        # gravity
        square_state["vy"] += GRAVITY
        square_state["y"] += square_state["vy"]

        # ground collision
        if square_state["y"] >= GROUND_Y:
            square_state["y"] = GROUND_Y
            square_state["vy"] = 0.0
            square_state["on_ground"] = True
        else:
            square_state["on_ground"] = False

        # horizontal bounds
        if square_state["x"] < 0:
            square_state["x"] = 0
        if square_state["x"] > self.width() - SQUARE_SIZE:
            square_state["x"] = self.width() - SQUARE_SIZE

        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        x = int(square_state["x"])
        y = int(square_state["y"])
        rect = QtCore.QRect(x, y, SQUARE_SIZE, SQUARE_SIZE)

        # red square, everything else transparent
        color = QtGui.QColor(255, 0, 0, 255)
        painter.fillRect(rect, color)


def start_overlay():
    """Start the transparent, click-through overlay on Windows."""
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

    # Make the overlay click-through (mouse goes to desktop/apps)
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

    print("[sender] Overlay started (gravity + jump)")
    app.exec_()


if __name__ == "__main__":
    # Run streaming and control server in background threads
    threading.Thread(target=screen_sender, daemon=True).start()
    threading.Thread(target=control_server, daemon=True).start()

    # Run the overlay GUI in the main thread
    start_overlay()
