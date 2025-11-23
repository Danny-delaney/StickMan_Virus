import socket
import struct
import threading

import cv2
import numpy as np

SENDER_IP = "127.0.0.1"
PORT = 5000          # video stream port
CONTROL_PORT = 5001  # control port

MOVE_STEP = 20       # pixels per key press

INITIAL_WIDTH = 960
INITIAL_HEIGHT = 540

# Shared frame between network thread and UI thread
current_frame = None
running = True
frame_lock = threading.Lock()


def recvall(sock, n):
    """Receive exactly n bytes or return None if the connection closes."""
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


def video_receiver(sock):
    """Background thread: receive frames over TCP and update current_frame."""
    global current_frame, running

    try:
        while running:
            header = recvall(sock, 4)
            if header is None:
                print("[receiver] Video connection closed by sender")
                break

            (frame_len,) = struct.unpack("!I", header)

            jpeg_data = recvall(sock, frame_len)
            if jpeg_data is None:
                print("[receiver] Video connection closed during frame")
                break

            arr = np.frombuffer(jpeg_data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            with frame_lock:
                current_frame = frame

    finally:
        running = False
        try:
            sock.close()
        except OSError:
            pass
        print("[receiver] Video thread exiting")


def main():
    global running

    # --- connect to video stream ---
    vid_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"[receiver] Connecting to {SENDER_IP}:{PORT} for video...")
    vid_sock.connect((SENDER_IP, PORT))
    print("[receiver] Connected for video")

    # --- connect to control channel ---
    ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"[receiver] Connecting to {SENDER_IP}:{CONTROL_PORT} for control...")
    ctrl_sock.connect((SENDER_IP, CONTROL_PORT))
    print("[receiver] Connected for control")

    def send_move(dx, dy):
        """Send a movement packet. Small and non-blocking-ish."""
        try:
            packet = struct.pack("!ii", dx, dy)
            # send() is fine for this tiny 8-byte packet
            ctrl_sock.send(packet)
        except OSError:
            pass

    # Start background video thread
    t = threading.Thread(target=video_receiver, args=(vid_sock,), daemon=True)
    t.start()

    window_name = "Remote Desktop"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, INITIAL_WIDTH, INITIAL_HEIGHT)

    try:
        while running:
            # Get latest frame (if any)
            with frame_lock:
                frame = None if current_frame is None else current_frame.copy()

            if frame is not None:
                fh, fw = frame.shape[:2]

                # Fit inside current window size
                try:
                    x, y, win_w, win_h = cv2.getWindowImageRect(window_name)
                except AttributeError:
                    win_w, win_h = INITIAL_WIDTH, INITIAL_HEIGHT

                if win_w > 0 and win_h > 0:
                    scale = min(win_w / fw, win_h / fh)
                    new_w = max(1, int(fw * scale))
                    new_h = max(1, int(fh * scale))

                    resized = cv2.resize(frame, (new_w, new_h))
                    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                    x_offset = (win_w - new_w) // 2
                    y_offset = (win_h - new_h) // 2
                    canvas[y_offset:y_offset + new_h,
                           x_offset:x_offset + new_w] = resized
                    cv2.imshow(window_name, canvas)
                else:
                    cv2.imshow(window_name, frame)

            # Keyboard handling is now independent of video receiving
            key = cv2.waitKey(1) & 0xFF

            if key == 27 or key == ord('q'):  # ESC or q
                print("[receiver] Quit key pressed, exiting")
                running = False
                break
            elif key == ord('w'):
                send_move(0, -MOVE_STEP)
            elif key == ord('s'):
                send_move(0, MOVE_STEP)
            elif key == ord('a'):
                send_move(-MOVE_STEP, 0)
            elif key == ord('d'):
                send_move(MOVE_STEP, 0)

    finally:
        running = False
        try:
            ctrl_sock.close()
        except OSError:
            pass
        cv2.destroyAllWindows()
        print("[receiver] Main loop exiting")


if __name__ == "__main__":
    main()
