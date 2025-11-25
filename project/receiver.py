import socket
import struct
import threading

import cv2
import numpy as np

SENDER_IP = "149.153.106.23"
PORT = 5000          # video stream port
CONTROL_PORT = 5001  # control port

MOVE_STEP = 20       # pixels per key press

INITIAL_WIDTH = 960
INITIAL_HEIGHT = 540


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


def video_receiver(sock, frame_lock, frame_holder, running_flag):
    """
    Receive JPEG frames from the sender.
    Protocol:
      - read 4 bytes: frame length, big endian
      - read frame bytes
    """
    while running_flag["running"]:
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
            frame_holder["frame"] = frame


def main():
    running_flag = {"running": True}
    frame_lock = threading.Lock()
    frame_holder = {"frame": None}

    vid_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"[receiver] Connecting to {SENDER_IP}:{PORT} for video...")
    vid_sock.connect((SENDER_IP, PORT))
    print("[receiver] Connected for video")

    # connect to control channel
    ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"[receiver] Connecting to {SENDER_IP}:{CONTROL_PORT} for control...")
    ctrl_sock.connect((SENDER_IP, CONTROL_PORT))
    print("[receiver] Connected for control")

    def send_move(dx, dy, action=0):
        """Send one control packet to the sender."""
        try:
            # dx, dy, action (0=move/jump, 1=punch/click)
            packet = struct.pack("!iii", int(dx), int(dy), int(action))
            ctrl_sock.sendall(packet)
        except OSError:
            pass

    # start background video thread
    t = threading.Thread(
        target=video_receiver,
        args=(vid_sock, frame_lock, frame_holder, running_flag),
        daemon=True,
    )
    t.start()

    window_name = "ReceiverVideo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, INITIAL_WIDTH, INITIAL_HEIGHT)

    print("[receiver] Controls: WASD to move, SPACE to punch/click, Q/ESC to quit")

    try:
        while running_flag["running"]:
            with frame_lock:
                frame = None if frame_holder["frame"] is None else frame_holder["frame"].copy()

            if frame is not None:
                fh, fw = frame.shape[:2]

                # scale frame to fit the current window
                try:
                    x, y, win_w, win_h = cv2.getWindowImageRect(window_name)
                except AttributeError:
                    win_w, win_h = INITIAL_WIDTH, INITIAL_HEIGHT

                if win_w > 0 and win_h > 0:
                    scale = min(win_w / fw, win_h / fh)
                    new_w = int(fw * scale)
                    new_h = int(fh * scale)
                    resized = cv2.resize(frame, (new_w, new_h))

                    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                    y_offset = (win_h - new_h) // 2
                    x_offset = (win_w - new_w) // 2
                    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
                    cv2.imshow(window_name, canvas)
                else:
                    cv2.imshow(window_name, frame)

            # keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord('q'):  # ESC or q
                print("[receiver] Quit key pressed, exiting")
                running_flag["running"] = False
                break
            elif key == ord('w'):
                send_move(0, -MOVE_STEP)
            elif key == ord('s'):
                send_move(0, MOVE_STEP)
            elif key == ord('a'):
                send_move(-MOVE_STEP, 0)
            elif key == ord('d'):
                send_move(MOVE_STEP, 0)
            elif key == 32:  # SPACE
                send_move(0, 0, action=1)

    finally:
        running_flag["running"] = False
        try:
            ctrl_sock.close()
        except OSError:
            pass
        cv2.destroyAllWindows()
        print("[receiver] Main loop exiting")


if __name__ == "__main__":
    main()
