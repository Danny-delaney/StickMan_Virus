import ctypes
import ctypes.wintypes
import sys

if sys.platform != "win32":
    raise RuntimeError("notepad_helper.py only works on Windows")

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

WM_SETTEXT = 0x000C
WM_GETTEXT = 0x000D
WM_GETTEXTLENGTH = 0x000E
EM_POSFROMCHAR = 0x00D6
EM_CHARFROMPOS = 0x00D7

class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

def find_notepad_edit_hwnd():
    main_hwnd = user32.FindWindowW("Notepad", None)
    if not main_hwnd:
        return None

    edit_hwnd = user32.FindWindowExW(main_hwnd, None, "RichEditD2DPT", None)
    if not edit_hwnd:
        edit_hwnd = user32.FindWindowExW(main_hwnd, None, "Edit", None)
    
    return edit_hwnd

def read_notepad():
    hwnd = find_notepad_edit_hwnd()
    if not hwnd:
        return "Notepad not found."

    length = user32.SendMessageW(hwnd, WM_GETTEXTLENGTH, 0, 0)
    if length == 0:
        return ""

    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.SendMessageW(hwnd, WM_GETTEXT, length + 1, ctypes.byref(buffer))
    return buffer.value

def write_notepad(text):
    hwnd = find_notepad_edit_hwnd()
    if not hwnd:
        return False
    
    result = user32.SendMessageW(hwnd, WM_SETTEXT, 0, ctypes.c_wchar_p(text))
    return bool(result)

def get_char_screen_pos(char_index):
    hwnd = find_notepad_edit_hwnd()
    if not hwnd:
        return None

    pos = user32.SendMessageW(hwnd, EM_POSFROMCHAR, char_index, 0)
    if pos == -1:
        return None

    x = pos & 0xFFFF
    y = (pos >> 16) & 0xFFFF
    
    pt = POINT(x, y)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    return (pt.x, pt.y)

def get_char_index_at_screen_pos(screen_x, screen_y):
    hwnd = find_notepad_edit_hwnd()
    if not hwnd: return None

    # Convert screen coords to Notepad coords
    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
    pt = POINT(int(screen_x), int(screen_y))
    user32.ScreenToClient(hwnd, ctypes.byref(pt))
    
    # Send message to get character index at that point
    lParam = (pt.y << 16) | (pt.x & 0xFFFF)
    res = user32.SendMessageW(hwnd, 0x00D7, 0, lParam) # 0x00D7 = EM_CHARFROMPOS
    return res & 0xFFFF if res != -1 else None

def append_notepad(text):
    current_text = read_notepad()
    return write_notepad(current_text + text)

if __name__ == "__main__":
    print("Reading Notepad...")
    content = read_notepad()
    print(f"Current Content: {content}")
    
    if len(content) > 0:
        pos = get_char_screen_pos(0)
        print(f"First character is at screen position: {pos}")