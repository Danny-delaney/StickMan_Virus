import socket
import struct

import cv2
import numpy as np

SENDER_IP = "127.0.0.1"
PORT = 5000

def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        print(f"[receiver] Connecting to {SENDER_IP}:{PORT}...")
        sock.connect((SENDER_IP, PORT))
        print("[receiver] Connected")

        window_name = "Remote Desktop"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        try:
            while True:
                header = recvall(sock, 4)
                if header is None:
                    print("[receiver] Connection closed by sender")
                    break

                (size,) = struct.unpack("!I", header)

                img_data = recvall(sock, size)
                if img_data is None:
                    print("[receiver] Failed to read frame data")
                    break

                arr = np.frombuffer(img_data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                fh, fw = frame.shape[:2]

                try:
                    x, y, win_w, win_h = cv2.getWindowImageRect(window_name)
                except AttributeError:
                    win_w, win_h = 1280, 720

                if win_w > 0 and win_h > 0:
                    scale = min(win_w / fw, win_h / fh)
                    new_w = max(1, int(fw * scale))
                    new_h = max(1, int(fh * scale))

                    resized = cv2.resize(frame, (new_w, new_h))

                    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                    x_offset = (win_w - new_w) // 2
                    y_offset = (win_h - new_h) // 2
                    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

                    cv2.imshow(window_name, canvas)
                else:
                    cv2.imshow(window_name, frame)

                if cv2.waitKey(1) == 27:
                    print("[receiver] ESC pressed, exiting")
                    break
        finally:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
