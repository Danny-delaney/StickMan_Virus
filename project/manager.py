import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import time
import os
import yaml
import ctypes

from window_platforms import (
    get_window_platforms,
    get_window_text,
    get_class_name,
)

user32 = ctypes.windll.user32

def set_window_pos(hwnd, x, y, w, h):
    SWP_NOZORDER = 0x0004
    user32.SetWindowPos(hwnd, 0, x, y, w, h, SWP_NOZORDER)

def minimize_all_except_apps():
    safe_list = [
        "ReceiverVideo",
        "PythonOverlay",
        "Launcher",
        "Layout Manager",
        "TkChild",
        "Toplevel"
    ]

    def enum_handler(hwnd, lparam):
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)
            title = buff.value

            class_buff = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, class_buff, 256)
            cls = class_buff.value

            is_safe = any(item.lower() in title.lower() or item.lower() in cls.lower() 
                         for item in safe_list)

            if not is_safe and not user32.IsIconic(hwnd):
                user32.ShowWindow(hwnd, 6)
        return True

    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    user32.EnumWindows(EnumWindowsProc(enum_handler), 0)

def find_window(criteria):
    for x, y, w, h, hwnd in get_window_platforms():
        title = get_window_text(hwnd)
        cls = get_class_name(hwnd)

        if criteria.get("title"):
            if criteria["title"].lower() not in title.lower():
                continue

        if criteria.get("class"):
            if criteria["class"] != cls:
                continue

        return hwnd

    return None


def is_window_present(criteria):
    return find_window(criteria) is not None

class LayoutApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Layout Manager")
        self.geometry("380x260")
        self.resizable(False, False)

        self.layouts = self.load_layouts()
        self.selected = tk.StringVar()

        self._build_ui()

    def _build_ui(self):
        ttk.Label(self, text="Select Layout", font=("Segoe UI", 12)).pack(pady=12)

        self.combo = ttk.Combobox(
            self,
            textvariable=self.selected,
            values=list(self.layouts.keys()),
            state="readonly"
        )
        self.combo.pack(pady=6, padx=20, fill="x")

        ttk.Button(self, text="Apply Layout", command=self.apply_layout).pack(pady=20)

    def load_layouts(self):
        path = os.path.join(os.path.dirname(__file__), "layouts.yaml")

        if not os.path.exists(path):
            messagebox.showerror("Error", "layouts.yaml not found")
            return {}

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if not data:
            messagebox.showerror("Error", "layouts.yaml is empty or invalid")
            return {}

        return data.get("layouts", {})

    def apply_layout(self):
        name = self.selected.get()

        if not name:
            messagebox.showwarning("No selection", "Pick a layout")
            return

        layout = self.layouts.get(name, [])

        minimize_all_except_apps()

        for win in layout:
            criteria = {
                "title": win.get("title"),
                "class": win.get("class"),
            }

            if not is_window_present(criteria):
                cmd = win.get("cmd")
                if cmd:
                    subprocess.Popen(cmd)

        time.sleep(2.5)

        for win in layout:
            hwnd = find_window({
                "title": win.get("title"),
                "class": win.get("class"),
            })

            if hwnd:
                set_window_pos(
                    hwnd,
                    win.get("x", 0),
                    win.get("y", 0),
                    win.get("width", 800),
                    win.get("height", 600),
                )
            else:
                print(f"[WARN] Window not found: {win.get('name', 'unknown')}")

if __name__ == "__main__":
    app = LayoutApp()
    app.mainloop()