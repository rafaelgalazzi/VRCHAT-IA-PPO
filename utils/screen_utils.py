# Screen capture utilities
import mss
import numpy as np
from PIL import Image
import win32gui
import win32con

def get_vrchat_window_bbox():
    hwnd = win32gui.FindWindow(None, "VRChat")
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

def capture_vrchat_frame():
    rect = get_vrchat_window_bbox()
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
