# Screen capture utilities
import mss
import numpy as np
from PIL import Image
import win32gui
import win32con
import win32process
import psutil



def find_window_by_process(exe_name):
    result = []

    def enum_handler(hwnd, _):
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            proc = psutil.Process(pid)
            if proc.name().lower() == exe_name.lower():
                result.append(hwnd)
        except:
            pass

    win32gui.EnumWindows(enum_handler, None)
    return result[0]


def get_window_bbox():
    print(find_window_by_process("vrchat.exe"))
    hwnd = find_window_by_process("vrchat.exe")
    if hwnd == 0:
        return None
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        rect = win32gui.GetWindowRect(hwnd)
        if rect[2] - rect[0] <= 0 or rect[3] - rect[1] <= 0:
            return None
        return rect
    except Exception:
        return None

def capture_window_frame():
    rect = get_window_bbox()
    if not rect:
        return None
    with mss.mss() as sct:
        monitor = {
            "top": rect[1],
            "left": rect[0],
            "width": rect[2] - rect[0],
            "height": rect[3] - rect[1]
        }
        sct_img = sct.grab(monitor)
        img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        return img
