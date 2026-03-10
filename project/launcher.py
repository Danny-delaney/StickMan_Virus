import os
import sys
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox

DEFAULT_IP = "127.0.0.1"


class LauncherApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Sender / Receiver Launcher")
        self.geometry("360x260")
        self.resizable(False, False)

        self.mode = tk.StringVar(value="receiver")
        self.ip = tk.StringVar(value=DEFAULT_IP)
        self.status = tk.StringVar(value="Ready")

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 14, "pady": 8}

        title = ttk.Label(self, text="Launcher", font=("Segoe UI", 14, "bold"))
        title.pack(pady=(14, 8))

        mode_frame = ttk.LabelFrame(self, text="Mode")
        mode_frame.pack(fill="x", padx=14, pady=6)

        ttk.Radiobutton(
            mode_frame, text="Receiver", variable=self.mode, value="receiver"
        ).pack(anchor="w", padx=10, pady=4)

        ttk.Radiobutton(
            mode_frame, text="Sender", variable=self.mode, value="sender"
        ).pack(anchor="w", padx=10, pady=4)

        ip_frame = ttk.Frame(self)
        ip_frame.pack(fill="x", **pad)

        ttk.Label(ip_frame, text="IP:").pack(side="left")
        ttk.Entry(ip_frame, textvariable=self.ip).pack(
            side="left", fill="x", expand=True, padx=(8, 0)
        )

        hint = ttk.Label(
            self,
            text="Use 127.0.0.1 for same PC",
            foreground="gray"
        )
        hint.pack(anchor="w", padx=14)

        button_frame = ttk.Frame(self)
        button_frame.pack(fill="x", padx=14, pady=14)

        ttk.Button(button_frame, text="Launch", command=self.launch).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(button_frame, text="Quit", command=self.destroy).pack(
            side="left", fill="x", expand=True, padx=(8, 0)
        )

        ttk.Label(self, textvariable=self.status).pack(anchor="w", padx=14, pady=(0, 8))

    def launch(self):
        ip = self.ip.get().strip() or DEFAULT_IP
        mode = self.mode.get()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        python_exe = sys.executable

        try:
            if mode == "receiver":
                script = os.path.join(base_dir, "receiver.py")
                cmd = [python_exe, script, "--ip", ip]
            else:
                script = os.path.join(base_dir, "sender.py")
                cmd = [python_exe, script, "--host", ip]

            subprocess.Popen(cmd, cwd=base_dir)
            self.status.set(f"Launched {mode} with {ip}")

        except Exception as e:
            messagebox.showerror("Launch failed", str(e))
            self.status.set("Launch failed")


if __name__ == "__main__":
    app = LauncherApp()
    app.mainloop()