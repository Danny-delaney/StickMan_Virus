import socket
import struct
import threading
import time

import cv2
import numpy as np

from pynput import keyboard

SENDER_IP = "127.0.0.1"
PORT = 5000
CONTROL_PORT = 5001

INITIAL_WIDTH = 960
INITIAL_HEIGHT = 540

CONTROL_HZ = 60
SPEED_PX_PER_SEC = 900


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


def video_receiver(sock, frame_lock, frame_holder, running_flag):
    base = None

    while running_flag["running"]:
        msg_type = recvall(sock, 1)
        if not msg_type:
            break

        t = msg_type
        if t == b"F":
            header = recvall(sock, 4)
            if not header:
                break
            (size,) = struct.unpack("!I", header)
            payload = recvall(sock, size)
            if not payload:
                break
            frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            with frame_lock:
                base = frame
                frame_holder["frame"] = base

        elif t == b"P":
            header = recvall(sock, 2)
            if not header:
                break
            (count,) = struct.unpack("!H", header)

            if base is None:
                for _ in range(count):
                    h2 = recvall(sock, 12)
                    if not h2:
                        running_flag["running"] = False
                        break
                    x, y, w, h, size = struct.unpack("!HHHHI", h2)
                    payload = recvall(sock, size)
                    if not payload:
                        running_flag["running"] = False
                        break
                continue

            with frame_lock:
                for _ in range(count):
                    h2 = recvall(sock, 12)
                    if not h2:
                        running_flag["running"] = False
                        break
                    x, y, w, h, size = struct.unpack("!HHHHI", h2)
                    payload = recvall(sock, size)
                    if not payload:
                        running_flag["running"] = False
                        break
                    patch = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if patch is None:
                        continue

                    H, W = base.shape[:2]
                    x2 = min(W, x + w)
                    y2 = min(H, y + h)
                    x = max(0, x)
                    y = max(0, y)
                    if x >= x2 or y >= y2:
                        continue

                    target_w = x2 - x
                    target_h = y2 - y
                    if patch.shape[1] != target_w or patch.shape[0] != target_h:
                        patch = cv2.resize(patch, (target_w, target_h))

                    base[y:y2, x:x2] = patch

                frame_holder["frame"] = base

        else:
            break

    running_flag["running"] = False


def main():
    running_flag = {"running": True}
    frame_lock = threading.Lock()
    frame_holder = {"frame": None}

    vid_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"[receiver] Connecting to {SENDER_IP}:{PORT} for video...")
    vid_sock.connect((SENDER_IP, PORT))
    print("[receiver] Connected for video")

    ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"[receiver] Connecting to {SENDER_IP}:{CONTROL_PORT} for control...")
    ctrl_sock.connect((SENDER_IP, CONTROL_PORT))
    print("[receiver] Connected for control")

    def send_move(dx, dy, action=0):
        try:
            packet = struct.pack("!iii", int(dx), int(dy), int(action))
            ctrl_sock.sendall(packet)
        except OSError:
            pass

    pressed = set()
    click_pending = {"v": False}
    quit_flag = {"v": False}

    def on_press(key):
        if not running_flag["running"]:
            return False
        try:
            k = key.char.lower()
            if k in ("w", "a", "s", "d"):
                pressed.add(k)
            elif k == "q":
                quit_flag["v"] = True
        except AttributeError:
            if key == keyboard.Key.space:
                click_pending["v"] = True
            elif key == keyboard.Key.esc:
                quit_flag["v"] = True

    def on_release(key):
        try:
            k = key.char.lower()
            if k in ("w", "a", "s", "d"):
                pressed.discard(k)
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()

    t = threading.Thread(
        target=video_receiver,
        args=(vid_sock, frame_lock, frame_holder, running_flag),
        daemon=True,
    )
    t.start()

    window_name = "ReceiverVideo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, INITIAL_WIDTH, INITIAL_HEIGHT)

    print("[receiver] Controls: hold WASD to move, SPACE to punch/click, Q/ESC to quit")

    last_send = time.perf_counter()
    send_period = 1.0 / CONTROL_HZ

    try:
        while running_flag["running"]:
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    running_flag["running"] = False
                    break
            except cv2.error:
                running_flag["running"] = False
                break

            with frame_lock:
                frame = None if frame_holder["frame"] is None else frame_holder["frame"].copy()

            if frame is not None:
                fh, fw = frame.shape[:2]
                try:
                    _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
                except Exception:
                    win_w, win_h = INITIAL_WIDTH, INITIAL_HEIGHT

                if win_w > 0 and win_h > 0:
                    scale = min(win_w / fw, win_h / fh)
                    new_w = max(1, int(fw * scale))
                    new_h = max(1, int(fh * scale))
                    resized = cv2.resize(frame, (new_w, new_h))

                    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                    y_off = (win_h - new_h) // 2
                    x_off = (win_w - new_w) // 2
                    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
                    cv2.imshow(window_name, canvas)
                else:
                    cv2.imshow(window_name, frame)

            cv2.waitKey(1)

            if quit_flag["v"]:
                running_flag["running"] = False
                break

            now = time.perf_counter()
            if now - last_send >= send_period:
                dt = now - last_send
                if dt > 0.1:
                    dt = 0.1

                dx_dir = (1 if "d" in pressed else 0) - (1 if "a" in pressed else 0)
                dy_dir = (1 if "s" in pressed else 0) - (1 if "w" in pressed else 0)

                dx = dx_dir * SPEED_PX_PER_SEC * dt
                dy = dy_dir * SPEED_PX_PER_SEC * dt

                action = 1 if click_pending["v"] else 0
                click_pending["v"] = False

                if dx_dir != 0 or dy_dir != 0 or action != 0:
                    send_move(dx, dy, action)

                last_send = now

    finally:
        running_flag["running"] = False
        try:
            listener.stop()
        except Exception:
            pass
        try:
            ctrl_sock.close()
        except OSError:
            pass
        try:
            vid_sock.close()
        except OSError:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("[receiver] Main loop exiting")


if __name__ == "__main__":
    main()
