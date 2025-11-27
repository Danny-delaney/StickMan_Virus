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
PORT = 5000
CONTROL_PORT = 5001

MOVE_SPEED = 8 * 5
GRAVITY = 1.0 * 5
JUMP_VELOCITY = -16.0 * 5

INPUT_HOLD_MS = 80
PUNCH_MS = 220

SPRITE_FOOT_Y = int(SPRITE_HEIGHT * (60.0 / 64.0))

GROUND_Y = None

_HEAD_SRC_WIDTH = 14
_FRAME_SRC_WIDTH = 64
HITBOX_HALF_WIDTH = int(SPRITE_WIDTH * _HEAD_SRC_WIDTH / (2 * _FRAME_SRC_WIDTH))

WINDOW_EXIT_COOLDOWN_MS = 500

latest_sprite_rgba = None

square_state = {
    "x": 100.0,
    "y": 100.0,
    "vy": 0.0,
    "on_ground": False,
    "on_wall": False,
    "facing": 1,
    "input_vx": 0.0,
    "input_ttl_ms": 0,
    "punch_ttl_ms": 0,
    "entered_window": None,
    "down_pressed": False,
    "down_ttl_ms": 0,
    "window_enter_time_ms": 0,
}

FPS = 18
JPEG_QUALITY = 55

KEYFRAME_EVERY_SEC = 3.0
DIFF_THRESH = 18
DILATE_K = 7
MIN_RECT_AREA = 2000
MAX_RECTS = 30
MAX_CHANGED_FRACTION = 0.25


def recvall(sock, n: int):
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


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


def capture_window(hwnd):
    if sys.platform != "win32":
        return None

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32

    rect = RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None

    width = rect.right - rect.left
    height = rect.bottom - rect.top

    if width <= 0 or height <= 0:
        return None

    hwndDC = user32.GetWindowDC(hwnd)
    mfcDC = gdi32.CreateCompatibleDC(hwndDC)
    saveBitMap = gdi32.CreateCompatibleBitmap(hwndDC, width, height)
    gdi32.SelectObject(mfcDC, saveBitMap)

    user32.PrintWindow(hwnd, mfcDC, 2)

    class BITMAPINFOHEADER(ctypes.Structure):
        _fields_ = [
            ("biSize", ctypes.c_uint32),
            ("biWidth", ctypes.c_int32),
            ("biHeight", ctypes.c_int32),
            ("biPlanes", ctypes.c_uint16),
            ("biBitCount", ctypes.c_uint16),
            ("biCompression", ctypes.c_uint32),
            ("biSizeImage", ctypes.c_uint32),
            ("biXPelsPerMeter", ctypes.c_int32),
            ("biYPelsPerMeter", ctypes.c_int32),
            ("biClrUsed", ctypes.c_uint32),
            ("biClrImportant", ctypes.c_uint32),
        ]

    class BITMAPINFO(ctypes.Structure):
        _fields_ = [
            ("bmiHeader", BITMAPINFOHEADER),
            ("bmiColors", ctypes.c_uint32 * 3),
        ]

    bmi = BITMAPINFO()
    bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
    bmi.bmiHeader.biWidth = width
    bmi.bmiHeader.biHeight = -height
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bmi.bmiHeader.biCompression = 0

    bmp_array = np.zeros((height, width, 4), dtype=np.uint8)

    gdi32.GetDIBits(
        mfcDC, saveBitMap, 0, height,
        bmp_array.ctypes.data_as(ctypes.c_void_p),
        ctypes.byref(bmi), 0
    )

    gdi32.DeleteObject(saveBitMap)
    gdi32.DeleteDC(mfcDC)
    user32.ReleaseDC(hwnd, hwndDC)

    frame = cv2.cvtColor(bmp_array, cv2.COLOR_BGRA2BGR)
    return frame


def _merge_rects(rects, gap=8):
    if not rects:
        return []
    merged = []
    for (x, y, w, h) in rects:
        placed = False
        for i in range(len(merged)):
            mx, my, mw, mh = merged[i]
            ax1, ay1 = mx - gap, my - gap
            ax2, ay2 = mx + mw + gap, my + mh + gap
            bx1, by1 = x, y
            bx2, by2 = x + w, y + h

            if not (bx2 < ax1 or bx1 > ax2 or by2 < ay1 or by1 > ay2):
                nx1 = min(mx, x)
                ny1 = min(my, y)
                nx2 = max(mx + mw, x + w)
                ny2 = max(my + mh, y + h)
                merged[i] = (nx1, ny1, nx2 - nx1, ny2 - ny1)
                placed = True
                break
        if not placed:
            merged.append((x, y, w, h))
    return merged


def _find_change_rects(prev_bgr, curr_bgr):
    diff = cv2.absdiff(curr_bgr, prev_bgr)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, DIFF_THRESH, 255, cv2.THRESH_BINARY)

    kernel = np.ones((DILATE_K, DILATE_K), dtype=np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < MIN_RECT_AREA:
            continue
        rects.append((x, y, w, h))

    rects = _merge_rects(rects, gap=8)
    rects.sort(key=lambda r: r[2] * r[3], reverse=True)
    if len(rects) > MAX_RECTS:
        rects = rects[:MAX_RECTS]

    return rects


def _send_full(conn, frame_bgr):
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        return False
    payload = encoded.tobytes()
    conn.sendall(b"F" + struct.pack("!I", len(payload)) + payload)
    return True


def _send_patches(conn, frame_bgr, rects):
    conn.sendall(b"P" + struct.pack("!H", len(rects)))
    for (x, y, w, h) in rects:
        patch = frame_bgr[y:y + h, x:x + w]
        ok, encoded = cv2.imencode(".jpg", patch, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            continue
        payload = encoded.tobytes()
        header = struct.pack("!HHHHI", x, y, w, h, len(payload))
        conn.sendall(header)
        conn.sendall(payload)


def screen_sender():
    global latest_sprite_rgba

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
                prev = None
                last_keyframe = 0.0
                period = 1.0 / FPS
                next_t = time.perf_counter()
                prev_entered_window = None

                while True:
                    now = time.perf_counter()
                    if now < next_t:
                        time.sleep(next_t - now)
                    next_t += period

                    entered_hwnd = square_state.get("entered_window")

                    if entered_hwnd is not None:
                        window_frame = capture_window(entered_hwnd)
                        if window_frame is None:
                            square_state["entered_window"] = None
                            square_state["window_enter_time_ms"] = 0
                            continue

                        user32 = ctypes.windll.user32
                        rect = RECT()
                        frame = window_frame.copy()
                        if user32.GetWindowRect(entered_hwnd, ctypes.byref(rect)):
                            win_x = rect.left
                            win_y = rect.top

                            sprite_x_in_window = int(square_state["x"] - win_x)
                            sprite_y_in_window = int(square_state["y"] - win_y)

                            sprite_rgba = latest_sprite_rgba
                            if sprite_rgba is not None:
                                sh, sw = sprite_rgba.shape[:2]
                                H, W = frame.shape[:2]

                                sx = sprite_x_in_window
                                sy = sprite_y_in_window

                                x1 = max(0, sx)
                                y1 = max(0, sy)
                                x2 = min(W, sx + sw)
                                y2 = min(H, sy + sh)

                                if x2 > x1 and y2 > y1:
                                    sub_sprite = sprite_rgba[y1 - sy:y2 - sy, x1 - sx:x2 - sx]
                                    sub_frame = frame[y1:y2, x1:x2]

                                    alpha = sub_sprite[:, :, 3:4].astype(np.float32) / 255.0
                                    rgb = sub_sprite[:, :, :3].astype(np.float32)
                                    base = sub_frame.astype(np.float32)

                                    blended = alpha * rgb + (1.0 - alpha) * base
                                    frame[y1:y2, x1:x2] = blended.astype(np.uint8)
                        else:
                            frame = window_frame

                        if entered_hwnd != prev_entered_window:
                            prev = None
                            prev_entered_window = entered_hwnd
                    else:
                        img = np.array(sct.grab(monitor))
                        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                        if prev_entered_window is not None:
                            prev = None
                            prev_entered_window = None

                    try:
                        t = time.time()
                        if prev is None or (t - last_keyframe) >= KEYFRAME_EVERY_SEC:
                            if not _send_full(conn, frame):
                                continue
                            prev = frame
                            last_keyframe = t
                            continue

                        rects = _find_change_rects(prev, frame)
                        if not rects:
                            prev = frame
                            continue

                        total_area = frame.shape[0] * frame.shape[1]
                        changed_area = sum(w * h for (_, _, w, h) in rects)

                        if (changed_area / float(total_area)) > MAX_CHANGED_FRACTION:
                            _send_full(conn, frame)
                            last_keyframe = t
                        else:
                            _send_patches(conn, frame, rects)

                        prev = frame

                    except (BrokenPipeError, ConnectionResetError, OSError):
                        print("[sender] Video connection closed")
                        break


def control_server():
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

                    if dx != 0:
                        square_state["input_vx"] = MOVE_SPEED if dx > 0 else -MOVE_SPEED
                        square_state["input_ttl_ms"] = INPUT_HOLD_MS
                        square_state["facing"] = 1 if dx > 0 else -1

                    if dy > 0:
                        square_state["down_pressed"] = True
                        square_state["down_ttl_ms"] = INPUT_HOLD_MS

                    if dy < 0 and (square_state.get("on_ground", False) or square_state.get("on_wall", False)):
                        square_state["vy"] = JUMP_VELOCITY
                        square_state["on_ground"] = False
                        square_state["on_wall"] = False

            except (ConnectionResetError, OSError):
                print("[sender] Control connection error")
            finally:
                print("[sender] Control server thread exiting")


class Overlay(QtWidgets.QWidget):
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
        GROUND_Y = self.screen_h - SPRITE_FOOT_Y - 10

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

        self.platforms = get_window_platforms()

        if square_state["input_ttl_ms"] > 0:
            square_state["input_ttl_ms"] = max(0, square_state["input_ttl_ms"] - dt_ms)
        else:
            square_state["input_vx"] = 0.0

        if square_state.get("punch_ttl_ms", 0) > 0:
            square_state["punch_ttl_ms"] = max(0, int(square_state["punch_ttl_ms"]) - dt_ms)

        if square_state.get("down_ttl_ms", 0) > 0:
            square_state["down_ttl_ms"] = max(0, square_state["down_ttl_ms"] - dt_ms)
            if square_state["down_ttl_ms"] == 0:
                square_state["down_pressed"] = False
        else:
            square_state["down_pressed"] = False

        prev_x = square_state["x"]
        square_state["x"] += float(square_state["input_vx"])

        prev_y = square_state["y"]
        square_state["vy"] += GRAVITY
        square_state["y"] += square_state["vy"]

        square_state["on_ground"] = False
        square_state["on_wall"] = False

        entering_window = False
        if square_state.get("entered_window") is None and square_state.get("down_pressed", False):
            entering_window = self.try_enter_window(prev_y)

        if square_state.get("entered_window") is not None:
            bounds = self.get_window_bounds(square_state["entered_window"])
            if bounds is None:
                square_state["entered_window"] = None
                square_state["window_offset_x"] = 0
                square_state["window_offset_y"] = 0
                square_state["window_enter_time_ms"] = 0
                bounds = (0, 0, self.screen_w, self.screen_h)

            bounds_x, bounds_y, bounds_w, bounds_h = bounds

            square_state["x"] = max(bounds_x, min(bounds_x + bounds_w - SPRITE_WIDTH, square_state["x"]))

            window_floor = bounds_y + bounds_h - SPRITE_FOOT_Y
            if square_state["y"] >= window_floor and square_state["vy"] >= 0:
                square_state["y"] = window_floor
                square_state["vy"] = 0
                square_state["on_ground"] = True

            if square_state["y"] <= bounds_y:
                square_state["y"] = bounds_y
                square_state["vy"] = max(0, square_state["vy"])

            self.bring_window_to_front(square_state["entered_window"])

            if square_state.get("down_pressed") and square_state.get("on_ground"):
                now_ms = int(time.time() * 1000)
                enter_ms = square_state.get("window_enter_time_ms", 0)
                if enter_ms == 0 or now_ms - enter_ms >= WINDOW_EXIT_COOLDOWN_MS:
                    square_state["entered_window"] = None
                    square_state["window_offset_x"] = 0
                    square_state["window_offset_y"] = 0
                    square_state["window_enter_time_ms"] = 0

        else:
            square_state["x"] = max(0, min(self.screen_w - SPRITE_WIDTH, square_state["x"]))

            if not entering_window:
                self.handle_vertical_collisions(prev_y)
                self.handle_horizontal_collisions(prev_x)

            if square_state["y"] >= GROUND_Y and square_state["vy"] >= 0:
                square_state["y"] = GROUND_Y
                square_state["vy"] = 0
                square_state["on_ground"] = True

        self.update()

    def try_enter_window(self, prev_y):
        sx = square_state["x"]
        sy = square_state["y"]
        prev_bottom = prev_y + SPRITE_FOOT_Y
        curr_bottom = sy + SPRITE_FOOT_Y

        hit_center = sx + SPRITE_WIDTH / 2.0
        hit_left = hit_center - HITBOX_HALF_WIDTH
        hit_right = hit_center + HITBOX_HALF_WIDTH

        for plat_data in self.platforms:
            px, py, pw, ph, hwnd = plat_data
            platform_top = py
            platform_left = px
            platform_right = px + pw

            if hit_right > platform_left and hit_left < platform_right:
                if prev_bottom <= platform_top and curr_bottom >= platform_top:
                    square_state["entered_window"] = hwnd

                    bounds = self.get_window_bounds(hwnd)
                    if bounds:
                        square_state["window_offset_x"] = bounds[0]
                        square_state["window_offset_y"] = bounds[1]

                    square_state["window_enter_time_ms"] = int(time.time() * 1000)

                    return True

        return False

    def get_window_bounds(self, hwnd):
        if sys.platform != "win32":
            return None

        user32 = ctypes.windll.user32
        rect = RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None

        x = rect.left
        y = rect.top
        w = rect.right - rect.left
        h = rect.bottom - rect.top

        return (x, y, w, h)

    def bring_window_to_front(self, hwnd):
        if sys.platform == "win32":
            user32 = ctypes.windll.user32
            user32.SetForegroundWindow(hwnd)

    def handle_vertical_collisions(self, prev_y):
        sx = square_state["x"]
        sy = square_state["y"]
        vy = square_state["vy"]

        if vy < 0:
            return

        sprite_bottom = sy + SPRITE_FOOT_Y
        prev_bottom = prev_y + SPRITE_FOOT_Y

        hit_center = sx + SPRITE_WIDTH / 2.0
        hit_left = hit_center - HITBOX_HALF_WIDTH
        hit_right = hit_center + HITBOX_HALF_WIDTH

        for plat_data in self.platforms:
            px, py, pw, ph = plat_data[:4]
            platform_top = py
            platform_left = px
            platform_right = px + pw

            if hit_right <= platform_left or hit_left >= platform_right:
                continue

            if prev_bottom <= platform_top <= sprite_bottom:
                square_state["y"] = platform_top - SPRITE_FOOT_Y
                square_state["vy"] = 0
                square_state["on_ground"] = True
                return

    def handle_horizontal_collisions(self, prev_x):
        sx = square_state["x"]
        sy = square_state["y"]

        sprite_top = sy
        sprite_bottom = sy + SPRITE_FOOT_Y

        curr_center = sx + SPRITE_WIDTH / 2.0
        curr_left = curr_center - HITBOX_HALF_WIDTH
        curr_right = curr_center + HITBOX_HALF_WIDTH

        prev_center = prev_x + SPRITE_WIDTH / 2.0
        prev_left = prev_center - HITBOX_HALF_WIDTH
        prev_right = prev_center + HITBOX_HALF_WIDTH

        for plat_data in self.platforms:
            px, py, pw, ph = plat_data[:4]
            platform_left = px
            platform_right = px + pw
            platform_top = py
            platform_bottom = py + ph

            if sprite_bottom <= platform_top or sprite_top >= platform_bottom:
                continue

            if sx > prev_x:
                if prev_right <= platform_left <= curr_right:
                    new_center = platform_left - HITBOX_HALF_WIDTH
                    square_state["x"] = new_center - SPRITE_WIDTH / 2.0
                    square_state["on_wall"] = True
                    return

            if sx < prev_x:
                if curr_left <= platform_right <= prev_left:
                    new_center = platform_right + HITBOX_HALF_WIDTH
                    square_state["x"] = new_center - SPRITE_WIDTH / 2.0
                    square_state["on_wall"] = True
                    return

    def paintEvent(self, event):
        global latest_sprite_rgba

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        if square_state.get("entered_window") is None:
            platform_color = QtGui.QColor(0, 255, 0, 80)
            for plat_data in self.platforms:
                px, py, pw, ph = plat_data[:4]
                rect = QtCore.QRect(int(px), int(py), int(pw), 5)
                painter.fillRect(rect, platform_color)

        draw_x = int(square_state["x"])
        draw_y = int(square_state["y"])

        moving = abs(square_state.get("input_vx", 0.0)) > 0.1
        facing = square_state.get("facing", 1)
        on_ground = bool(square_state.get("on_ground", False))
        vy = float(square_state.get("vy", 0.0))
        punching = bool(square_state.get("punch_ttl_ms", 0) > 0)

        sprite_img = QtGui.QImage(
            SPRITE_WIDTH,
            SPRITE_HEIGHT,
            QtGui.QImage.Format_ARGB32_Premultiplied,
        )
        sprite_img.fill(QtCore.Qt.transparent)

        sp_painter = QtGui.QPainter(sprite_img)
        sp_painter.setRenderHint(QtGui.QPainter.Antialiasing)
        sp_painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        self.sprite.draw(
            sp_painter,
            0,
            0,
            facing=facing,
            moving=moving,
            on_ground=on_ground,
            vy=vy,
            punching=punching,
        )
        sp_painter.end()

        painter.drawImage(draw_x, draw_y, sprite_img)

        ptr = sprite_img.bits()
        ptr.setsize(sprite_img.height() * sprite_img.bytesPerLine())
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(
            (sprite_img.height(), sprite_img.bytesPerLine() // 4, 4)
        )
        arr = arr[:, :sprite_img.width(), :].copy()
        latest_sprite_rgba = arr


def start_overlay():
    app = QtWidgets.QApplication(sys.argv)
    overlay = Overlay()

    screen = QtWidgets.QApplication.primaryScreen()
    geo = screen.geometry()
    overlay.setGeometry(geo)

    overlay.show()

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


def left_click():
    if sys.platform != "win32":
        return
    
    hwnd = square_state.get("entered_window")
    if hwnd is None:
        return
    
    user32 = ctypes.windll.user32
    rect = RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return
    
    sx = int(square_state.get("x", 0))
    sy = int(square_state.get("y", 0))
    facing = 1 if square_state.get("facing", 1) >= 0 else -1
    
    if facing > 0:
        punch_x = sx + SPRITE_WIDTH
    else:
        punch_x = sx
    punch_y = sy + SPRITE_HEIGHT // 3
    
    client_x = punch_x - rect.left
    client_y = punch_y - rect.top
    
    lParam = (client_y << 16) | (client_x & 0xFFFF)
    
    WM_LBUTTONDOWN = 0x0201
    WM_LBUTTONUP = 0x0202
    
    user32.SendMessageW(hwnd, WM_LBUTTONDOWN, 0, lParam)
    user32.SendMessageW(hwnd, WM_LBUTTONUP, 0, lParam)


if __name__ == "__main__":
    threading.Thread(target=screen_sender, daemon=True).start()
    threading.Thread(target=control_server, daemon=True).start()
    start_overlay()
