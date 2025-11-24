"""
Find top-level windows on Windows and return their on-screen rectangles.
These rectangles are used as platforms for the square.
"""

import sys
import ctypes

if sys.platform != "win32":
    raise RuntimeError("window_platforms.py only works on Windows")

user32 = ctypes.windll.user32


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


def get_window_text(hwnd):
    """Get the window title."""
    length = user32.GetWindowTextLengthW(hwnd)
    if length == 0:
        return ""
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def get_class_name(hwnd):
    """Get the window class name."""
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def get_window_platforms(min_width=80, min_height=40):
    """
    Return a list of rectangles (x, y, w, h) for top-level windows.

    Each rectangle is clipped to the main screen.
    We skip hidden, minimized, tiny, desktop/taskbar, our overlay,
    and almost full-screen windows.
    """
    platforms = []

    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    IsWindowVisible = user32.IsWindowVisible
    IsIconic = user32.IsIconic
    GetWindowRect = user32.GetWindowRect

    desktop_w = user32.GetSystemMetrics(0)
    desktop_h = user32.GetSystemMetrics(1)
    screen_area = desktop_w * desktop_h

    def callback(hwnd, lParam):
        if not IsWindowVisible(hwnd):
            return True
        if IsIconic(hwnd):
            return True

        class_name = get_class_name(hwnd)
        title = get_window_text(hwnd)

        # ignore desktop, helper windows, taskbar, and our overlay
        if class_name in ("Progman", "WorkerW", "Shell_TrayWnd"):
            return True
        if title == "PythonOverlay":
            return True

        rect = RECT()
        if not GetWindowRect(hwnd, ctypes.byref(rect)):
            return True

        left, top, right, bottom = rect.left, rect.top, rect.right, rect.bottom

        # keep only the part inside the main screen
        clip_left = max(0, min(left, desktop_w))
        clip_top = max(0, min(top, desktop_h))
        clip_right = max(0, min(right, desktop_w))
        clip_bottom = max(0, min(bottom, desktop_h))

        width = clip_right - clip_left
        height = clip_bottom - clip_top

        if width < min_width or height < min_height:
            return True
        if clip_right <= 0 or clip_bottom <= 0:
            return True
        if clip_left >= desktop_w or clip_top >= desktop_h:
            return True

        # skip windows that cover almost the whole screen
        area = width * height
        if area >= 0.9 * screen_area:
            return True

        platforms.append((clip_left, clip_top, width, height))
        return True

    enum_cb = WNDENUMPROC(callback)
    user32.EnumWindows(enum_cb, 0)

    return platforms


if __name__ == "__main__":
    for p in get_window_platforms():
        print(p)
