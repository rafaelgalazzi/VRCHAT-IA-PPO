import os
import csv
import uuid
import time
import threading
from queue import Queue, Empty
from PIL import ImageGrab
from pynput import keyboard
import win32gui
import win32con
import ctypes
from ctypes import wintypes

# === CONFIGURAÇÕES ===
DATA_DIR = "data/images"
LABEL_FILE = "data/labels.csv"
FPS = 20
INTERVAL = 1.0 / FPS
IMAGE_FORMAT = "jpeg"  # ou "png"
JPEG_QUALITY = 90

# === ESTADO GLOBAL ===
pressed_keys = set()
mouse_dx = 0
mouse_dy = 0
recording = [True]
paused = [False]
frame_queue = Queue()
label_queue = Queue()

# === DEFINIÇÕES RAW INPUT ===
WM_INPUT = 0x00FF
RID_INPUT = 0x10000003
RIM_TYPEMOUSE = 0

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
wintypes.HCURSOR = wintypes.HANDLE
wintypes.HICON = wintypes.HANDLE
wintypes.HWND = wintypes.HANDLE
wintypes.HINSTANCE = wintypes.HANDLE
wintypes.HMENU = wintypes.HANDLE
wintypes.LRESULT = ctypes.c_long

class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [("usUsagePage", wintypes.USHORT),
                ("usUsage", wintypes.USHORT),
                ("dwFlags", wintypes.DWORD),
                ("hwndTarget", wintypes.HWND)]

class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [("dwType", wintypes.DWORD),
                ("dwSize", wintypes.DWORD),
                ("hDevice", wintypes.HANDLE),
                ("wParam", wintypes.WPARAM)]

class RAWMOUSE(ctypes.Structure):
    _fields_ = [("usFlags", wintypes.USHORT),
                ("usButtonFlags", wintypes.USHORT),
                ("usButtonData", wintypes.USHORT),
                ("ulRawButtons", wintypes.ULONG),
                ("lLastX", wintypes.LONG),
                ("lLastY", wintypes.LONG),
                ("ulExtraInformation", wintypes.ULONG)]

class RAWINPUTUNION(ctypes.Union):
    _fields_ = [("mouse", RAWMOUSE),
                ("keyboard", wintypes.BYTE * 24),
                ("hid", wintypes.BYTE * 24)]

class RAWINPUT(ctypes.Structure):
    _fields_ = [("header", RAWINPUTHEADER),
                ("data", RAWINPUTUNION)]

# WNDPROC corretamente tipada
WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_long, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.DefWindowProcW.restype = ctypes.c_long

def wnd_proc(hwnd, msg, wparam, lparam):
    global mouse_dx, mouse_dy
    if msg == WM_INPUT:
        dwSize = wintypes.UINT()
        user32.GetRawInputData(lparam, RID_INPUT, None, ctypes.byref(dwSize), ctypes.sizeof(RAWINPUTHEADER))
        buf = ctypes.create_string_buffer(dwSize.value)
        user32.GetRawInputData(lparam, RID_INPUT, buf, ctypes.byref(dwSize), ctypes.sizeof(RAWINPUTHEADER))
        raw = ctypes.cast(buf, ctypes.POINTER(RAWINPUT)).contents
        if raw.header.dwType == RIM_TYPEMOUSE:
            mouse_dx += raw.data.mouse.lLastX
            mouse_dy += raw.data.mouse.lLastY
    return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

def start_raw_input_listener():
    class WNDCLASS(ctypes.Structure):
        _fields_ = [("style", wintypes.UINT),
                    ("lpfnWndProc", WNDPROC),
                    ("cbClsExtra", ctypes.c_int),
                    ("cbWndExtra", ctypes.c_int),
                    ("hInstance", wintypes.HINSTANCE),
                    ("hIcon", wintypes.HICON),
                    ("hCursor", wintypes.HCURSOR),
                    ("hbrBackground", wintypes.HBRUSH),
                    ("lpszMenuName", wintypes.LPCWSTR),
                    ("lpszClassName", wintypes.LPCWSTR)]

    hInstance = kernel32.GetModuleHandleW(None)
    className = "RawInputCapture"

    wnd_class = WNDCLASS()
    wnd_class.lpfnWndProc = WNDPROC(wnd_proc)
    wnd_class.hInstance = hInstance
    wnd_class.lpszClassName = className

    if not user32.RegisterClassW(ctypes.byref(wnd_class)):
        raise ctypes.WinError()

    hwnd = user32.CreateWindowExW(0, className, className, 0,
                                  0, 0, 0, 0,
                                  None, None, hInstance, None)

    rid = RAWINPUTDEVICE(0x01, 0x02, 0x00000100, hwnd)  # Mouse
    if not user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(rid)):
        raise ctypes.WinError()

    msg = wintypes.MSG()
    while recording[0]:
        if user32.GetMessageW(ctypes.byref(msg), hwnd, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

def on_press(key):
    try:
        if hasattr(key, 'char') and key.char:
            pressed_keys.add(key.char.lower())
            if key.char.lower() == 'p':
                paused[0] = not paused[0]
                print("[PAUSA]" if paused[0] else "[RESUMIDO]")
        elif hasattr(key, 'name'):
            pressed_keys.add(key.name)
    except:
        pass

def on_release(key):
    try:
        if hasattr(key, 'char') and key.char:
            pressed_keys.discard(key.char.lower())
        elif hasattr(key, 'name'):
            pressed_keys.discard(key.name)
        if key == keyboard.Key.esc:
            recording[0] = False
            return False
    except:
        pass

def get_vrchat_window_bbox():
    hwnd = win32gui.FindWindow(None, "VRChat")
    if hwnd:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        return win32gui.GetWindowRect(hwnd)
    return None

def save_label(image_name, keys, dx, dy, timestamp):
    with open(LABEL_FILE, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([image_name, "+".join(keys), dx, dy, f"{timestamp:.3f}"])

def image_worker():
    while recording[0] or not frame_queue.empty():
        try:
            img, path = frame_queue.get(timeout=0.1)
            if IMAGE_FORMAT == "jpeg":
                img.convert("RGB").save(path, format="JPEG", quality=JPEG_QUALITY)
            else:
                img.save(path, format="PNG")
            frame_queue.task_done()
        except Empty:
            continue

def label_worker():
    while recording[0] or not label_queue.empty():
        try:
            args = label_queue.get(timeout=0.1)
            save_label(*args)
            label_queue.task_done()
        except Empty:
            continue

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(LABEL_FILE):
        with open(LABEL_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["image", "keys", "mouse_dx", "mouse_dy", "timestamp"])

    rect = get_vrchat_window_bbox()
    if not rect:
        print("[ERRO] Janela do VRChat não encontrada.")
        return

    print("Iniciando gravação. Pressione ESC para sair, P para pausar/resumir.")

    threading.Thread(target=start_raw_input_listener, daemon=True).start()
    threading.Thread(target=keyboard.Listener(on_press=on_press, on_release=on_release).start, daemon=True).start()

    for _ in range(2):
        threading.Thread(target=image_worker, daemon=True).start()
        threading.Thread(target=label_worker, daemon=True).start()

    global mouse_dx, mouse_dy
    start_global = time.time()

    while recording[0]:
        if paused[0]:
            time.sleep(0.1)
            continue

        t0 = time.time()
        now = time.time()
        timestamp = now - start_global

        img = ImageGrab.grab(bbox=rect).resize((224, 224))
        filename = f"{uuid.uuid4()}.{IMAGE_FORMAT}"
        path = os.path.join(DATA_DIR, filename)

        frame_queue.put((img, path))
        label_queue.put((filename, pressed_keys.copy(), mouse_dx, mouse_dy, timestamp))

        mouse_dx = 0
        mouse_dy = 0

        elapsed = time.time() - t0
        if elapsed < INTERVAL:
            time.sleep(INTERVAL - elapsed)

    print("Finalizando gravação...")
    frame_queue.join()
    label_queue.join()
    print("Gravação finalizada.")

if __name__ == "__main__":
    main()
