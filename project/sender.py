import socket
import struct
import time

import cv2
import mss
import numpy as np

HOST = "0.0.0.0"
PORT = 5000

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((HOST, PORT))
        server_sock.listen(1)

        print(f"[sender] Waiting for connection on {HOST}:{PORT}...")
        conn, addr = server_sock.accept()
        print(f"[sender] Connected by {addr}")

        with conn, mss.mss() as sct:
            monitor = sct.monitors[1]
            width = monitor["width"]
            height = monitor["height"]
            print(f"[sender] Capturing full display: {width}x{height}")

            try:
                while True:
                    sct_img = sct.grab(monitor)
                    frame = np.array(sct_img)

                    if frame.shape[2] == 4:
                        frame = frame[:, :, :3]

                    success, encoded = cv2.imencode(
                        ".jpg",
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 50],
                    )

                    if not success:
                        continue

                    data = encoded.tobytes()
                    size = len(data)

                    header = struct.pack("!I", size)
                    conn.sendall(header + data)

                    time.sleep(0.05)

            except (BrokenPipeError, ConnectionResetError):
                print("[sender] Connection closed by receiver")
            finally:
                print("[sender] Shutting down")

if __name__ == "__main__":
    main()
