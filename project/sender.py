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

MOVE_SPEED = 8 * 5
GRAVITY = 1.0 * 5
JUMP_VELOCITY = -16.0 * 5

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
    "entered_window": None,  # hwnd of window stickman is inside
    "down_pressed": False,   # track if down is being pressed
}

# --- Delta stream tuning ---
FPS = 30
JPEG_QUALITY = 70

KEYFRAME_EVERY_SEC = 2.0         # send a full frame at least this often
DIFF_THRESH = 18                  # pixel-diff threshold
DILATE_K = 7                      # merge nearby changes
MIN_RECT_AREA = 900               # ignore tiny noise
MAX_RECTS = 60                    # avoid too many patches
MAX_CHANGED_FRACTION = 0.35       # if too much changed, just send full frame


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


def capture_window(hwnd):
    """Capture a specific window by its handle, including off-screen parts."""
    if sys.platform != "win32":
        return None
    
    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    
    # Get window rect (may extend off screen)
    rect = RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    
    width = rect.right - rect.left
    height = rect.bottom - rect.top
    
    if width <= 0 or height <= 0:
        return None
    
    # Get window DC
    hwndDC = user32.GetWindowDC(hwnd)
    mfcDC = gdi32.CreateCompatibleDC(hwndDC)
    saveBitMap = gdi32.CreateCompatibleBitmap(hwndDC, width, height)
    gdi32.SelectObject(mfcDC, saveBitMap)
    
    # Print window to DC
    user32.PrintWindow(hwnd, mfcDC, 2)
    
    # Setup bitmap info structure
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
    bmi.bmiHeader.biHeight = -height  # negative for top-down
    bmi.bmiHeader.biPlanes = 1
    bmi.bmiHeader.biBitCount = 32
    bmi.bmiHeader.biCompression = 0  # BI_RGB
    
    # Create numpy array
    bmp_array = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Get the bits
    gdi32.GetDIBits(
        mfcDC, saveBitMap, 0, height,
        bmp_array.ctypes.data_as(ctypes.c_void_p),
        ctypes.byref(bmi), 0
    )
    
    # Clean up
    gdi32.DeleteObject(saveBitMap)
    gdi32.DeleteDC(mfcDC)
    user32.ReleaseDC(hwnd, hwndDC)
    
    # Convert BGRA to BGR
    frame = cv2.cvtColor(bmp_array, cv2.COLOR_BGRA2BGR)
    return frame


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


def _merge_rects(rects, gap=8):
    """Very simple O(n^2) merge of overlapping/nearby rectangles."""
    if not rects:
        return []
    merged = []
    for (x, y, w, h) in rects:
        placed = False
        for i in range(len(merged)):
            mx, my, mw, mh = merged[i]
            # expand merged rect by gap and check overlap
            ax1, ay1 = mx - gap, my - gap
            ax2, ay2 = mx + mw + gap, my + mh + gap
            bx1, by1 = x, y
            bx2, by2 = x + w, y + h

            if not (bx2 < ax1 or bx1 > ax2 or by2 < ay1 or by1 > ay2):
                # merge into merged[i]
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

    # merge and limit
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
    # Type 'F' + uint32 length + payload
    conn.sendall(b"F" + struct.pack("!I", len(payload)) + payload)
    return True


def _send_patches(conn, frame_bgr, rects):
    # Type 'P' + uint16 count + [x,y,w,h uint16 + uint32 len + bytes]...
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
    """
    Stream monitor using:
      - 'F' keyframes: full JPEG
      - 'P' patch frames: rectangles + JPEG for each rect
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
                prev = None
                last_keyframe = 0.0
                period = 1.0 / FPS
                next_t = time.perf_counter()
                prev_entered_window = None

                while True:
                    # simple FPS cap
                    now = time.perf_counter()
                    if now < next_t:
                        time.sleep(next_t - now)
                    next_t += period

                    entered_hwnd = square_state.get("entered_window")
                    
                    # If entered a window, capture just that window
                    if entered_hwnd is not None:
                        window_frame = capture_window(entered_hwnd)
                        if window_frame is None:
                            # Window closed or invalid, exit window mode
                            square_state["entered_window"] = None
                            continue
                        
                        # Get window position to draw sprite on the captured window
                        user32 = ctypes.windll.user32
                        rect = RECT()
                        if user32.GetWindowRect(entered_hwnd, ctypes.byref(rect)):
                            win_x = rect.left
                            win_y = rect.top
                            
                            # Convert stickman screen coords to window-relative coords
                            sprite_x_in_window = int(square_state["x"] - win_x)
                            sprite_y_in_window = int(square_state["y"] - win_y)
                            
                            # Draw stickman on the window capture
                            # Create a copy to draw on
                            frame = window_frame.copy()
                            
                            # For now, just use the window frame
                            # The stickman will be visible in the overlay on sender side
                            # and we need to composite it into the captured frame
                            frame = window_frame
                        else:
                            frame = window_frame
                        
                        # Force keyframe when entering/exiting window
                        if entered_hwnd != prev_entered_window:
                            prev = None
                            prev_entered_window = entered_hwnd
                    else:
                        # Normal full screen capture
                        img = np.array(sct.grab(monitor))
                        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                        
                        # Force keyframe when exiting window mode
                        if prev_entered_window is not None:
                            prev = None
                            prev_entered_window = None

                    try:
                        # Always send a keyframe at start, and periodically thereafter.
                        t = time.time()
                        if prev is None or (t - last_keyframe) >= KEYFRAME_EVERY_SEC:
                            if not _send_full(conn, frame):
                                continue
                            prev = frame
                            last_keyframe = t
                            continue

                        rects = _find_change_rects(prev, frame)
                        if not rects:
                            # nothing changed enough; don't send anything this tick
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
                    if dx != 0:
                        square_state["input_vx"] = MOVE_SPEED if dx > 0 else -MOVE_SPEED
                        square_state["input_ttl_ms"] = INPUT_HOLD_MS
                        square_state["facing"] = 1 if dx > 0 else -1

                    # Track down press state
                    square_state["down_pressed"] = (dy > 0)

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

        self.platforms = get_window_platforms()

        if square_state["input_ttl_ms"] > 0:
            square_state["input_ttl_ms"] = max(0, square_state["input_ttl_ms"] - dt_ms)
        else:
            square_state["input_vx"] = 0.0

        if square_state.get("punch_ttl_ms", 0) > 0:
            square_state["punch_ttl_ms"] = max(0, int(square_state["punch_ttl_ms"]) - dt_ms)

        prev_x = square_state["x"]
        square_state["x"] += float(square_state["input_vx"])
        
        prev_y = square_state["y"]
        square_state["vy"] += GRAVITY
        square_state["y"] += square_state["vy"]

        square_state["on_ground"] = False
        square_state["on_wall"] = False

        # Check if stickman wants to enter a window (before collision detection)
        entering_window = False
        if square_state.get("entered_window") is None and square_state.get("down_pressed", False):
            entering_window = self.try_enter_window(prev_y)

        # Get boundaries and handle movement based on mode
        if square_state.get("entered_window") is not None:
            # In window mode - coordinates are still screen-absolute
            bounds = self.get_window_bounds(square_state["entered_window"])
            if bounds is None:
                # Window no longer valid, exit window mode
                square_state["entered_window"] = None
                square_state["window_offset_x"] = 0
                square_state["window_offset_y"] = 0
                bounds = (0, 0, self.screen_w, self.screen_h)
            
            bounds_x, bounds_y, bounds_w, bounds_h = bounds
            
            # Clamp to window boundaries (screen coordinates)
            square_state["x"] = max(bounds_x, min(bounds_x + bounds_w - SPRITE_WIDTH, square_state["x"]))
            
            # Floor of window
            window_floor = bounds_y + bounds_h - SPRITE_HEIGHT
            if square_state["y"] >= window_floor and square_state["vy"] >= 0:
                square_state["y"] = window_floor
                square_state["vy"] = 0
                square_state["on_ground"] = True
            
            # Ceiling of window
            if square_state["y"] <= bounds_y:
                square_state["y"] = bounds_y
                square_state["vy"] = max(0, square_state["vy"])
            
            # Bring window to front
            self.bring_window_to_front(square_state["entered_window"])

            # --- New: hold DOWN while on the window floor to exit window mode ---
            if square_state.get("down_pressed") and square_state.get("on_ground"):
                print(f"[sender] Exiting window {square_state['entered_window']}")
                square_state["entered_window"] = None
                square_state["window_offset_x"] = 0
                square_state["window_offset_y"] = 0

        else:
            # Normal mode
            square_state["x"] = max(0, min(self.screen_w - SPRITE_WIDTH, square_state["x"]))
            
            # Only do collision detection if not entering a window
            if not entering_window:
                self.handle_vertical_collisions(prev_y)
                self.handle_horizontal_collisions(prev_x)
            
            # Normal ground collision
            if square_state["y"] >= GROUND_Y and square_state["vy"] >= 0:
                square_state["y"] = GROUND_Y
                square_state["vy"] = 0
                square_state["on_ground"] = True

        self.update()

    def try_enter_window(self, prev_y):
        """Check if stickman is standing on a platform and should drop through it."""
        sx = square_state["x"]
        sy = square_state["y"]
        prev_bottom = prev_y + SPRITE_HEIGHT
        curr_bottom = sy + SPRITE_HEIGHT
        sprite_left = sx
        sprite_right = sx + SPRITE_WIDTH

        for plat_data in self.platforms:
            px, py, pw, ph, hwnd = plat_data
            platform_top = py

            # Check if stickman is overlapping this platform horizontally
            if sprite_right > px and sprite_left < px + pw:
                # Check if stickman was standing on this platform (within collision range)
                if prev_bottom <= platform_top and curr_bottom >= platform_top:
                    square_state["entered_window"] = hwnd
                    
                    # Store the window bounds for reference
                    bounds = self.get_window_bounds(hwnd)
                    if bounds:
                        square_state["window_offset_x"] = bounds[0]
                        square_state["window_offset_y"] = bounds[1]
                    
                    print(f"[sender] Entered window {hwnd} at position ({sx}, {sy})")
                    # Allow the stickman to fall through by not stopping vertical movement
                    return True
        
        return False
    
    def get_window_bounds(self, hwnd):
        """Get the full bounds of a window (including off-screen parts)."""
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
        """Bring the specified window to the front."""
        if sys.platform == "win32":
            user32 = ctypes.windll.user32
            user32.SetForegroundWindow(hwnd)

    def handle_vertical_collisions(self, prev_y):
        sx = square_state["x"]
        sy = square_state["y"]
        vy = square_state["vy"]

        if vy < 0:
            return

        sprite_bottom = sy + SPRITE_HEIGHT
        prev_bottom = prev_y + SPRITE_HEIGHT

        for plat_data in self.platforms:
            px, py, pw, ph = plat_data[:4]
            platform_top = py
            sprite_left = sx
            sprite_right = sx + SPRITE_WIDTH

            if sprite_right <= px or sprite_left >= px + pw:
                continue

            if prev_bottom <= platform_top <= sprite_bottom:
                square_state["y"] = platform_top - SPRITE_HEIGHT
                square_state["vy"] = 0
                square_state["on_ground"] = True
                return

    def handle_horizontal_collisions(self, prev_x):
        sx = square_state["x"]
        sy = square_state["y"]

        sprite_top = sy
        sprite_bottom = sy + SPRITE_HEIGHT

        for plat_data in self.platforms:
            px, py, pw, ph = plat_data[:4]
            platform_left = px
            platform_right = px + pw
            platform_top = py
            platform_bottom = py + ph

            if sprite_bottom <= platform_top or sprite_top >= platform_bottom:
                continue

            if sx > prev_x:
                prev_right = prev_x + SPRITE_WIDTH
                curr_right = sx + SPRITE_WIDTH
                if prev_right <= platform_left <= curr_right:
                    square_state["x"] = platform_left - SPRITE_WIDTH
                    square_state["on_wall"] = True
                    return

            if sx < prev_x:
                prev_left = prev_x
                curr_left = sx
                if curr_left <= platform_right <= prev_left:
                    square_state["x"] = platform_right
                    square_state["on_wall"] = True
                    return

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Only draw platforms if not in a window
        if square_state.get("entered_window") is None:
            platform_color = QtGui.QColor(0, 255, 0, 80)
            for plat_data in self.platforms:
                px, py, pw, ph = plat_data[:4]
                rect = QtCore.QRect(int(px), int(py), int(pw), 5)
                painter.fillRect(rect, platform_color)

        # Draw stickman (always in screen coordinates for overlay)
        draw_x = int(square_state["x"])
        draw_y = int(square_state["y"])

        moving = abs(square_state.get("input_vx", 0.0)) > 0.1
        self.sprite.draw(
            painter,
            draw_x,
            draw_y,
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
    """Send one OS left-click (Windows only) at the end of the stickman's fist, then restore mouse position."""
    if sys.platform != "win32":
        return
    sx = int(square_state.get("x", 0))
    sy = int(square_state.get("y", 0))
    facing = 1 if square_state.get("facing", 1) >= 0 else -1
    user32 = ctypes.windll.user32
    desktop_w = int(user32.GetSystemMetrics(0))
    desktop_h = int(user32.GetSystemMetrics(1))
    if facing > 0:
        tx = sx + SPRITE_WIDTH
    else:
        tx = sx
    ty = sy + SPRITE_HEIGHT // 3
    tx = max(0, min(desktop_w - 1, tx))
    ty = max(0, min(desktop_h - 1, ty))

    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
    pt = POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    orig_x, orig_y = pt.x, pt.y

    user32.SetCursorPos(tx, ty)
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    user32.SetCursorPos(orig_x, orig_y)


if __name__ == "__main__":
    threading.Thread(target=screen_sender, daemon=True).start()
    threading.Thread(target=control_server, daemon=True).start()
    start_overlay()
