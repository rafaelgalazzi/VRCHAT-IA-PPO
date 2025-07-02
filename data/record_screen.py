import os
import csv
import uuid
import time
import threading
from PIL import ImageGrab
from pynput import keyboard
import win32gui
import win32con
import ctypes
from ctypes import wintypes
from queue import Queue, Empty

# Diretórios
DATA_DIR = "data/images"
LABEL_FILE = "data/labels.csv"
FPS = 20
INTERVAL = 1.0 / FPS
NUM_WORKER_THREADS = 2

# Configuração de formato de imagem: "png" ou "jpeg"
IMAGE_FORMAT = "jpeg"  # ou "png"
JPEG_QUALITY = 85      # Apenas se IMAGE_FORMAT == "jpeg"

# Estado
pressed_keys = set()
mouse_dx = 0
mouse_dy = 0
recording = [True]
is_paused = [False]
vrchat_hwnd = None

# Filas
image_queue = Queue()
label_queue = Queue()

WM_INPUT = 0x00FF

user32 = ctypes.WinDLL('user32')
kernel32 = ctypes.WinDLL('kernel32')

kernel32.GetModuleHandleW.restype = wintypes.HINSTANCE
kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]

user32.CreateWindowExW.restype = wintypes.HWND
user32.CreateWindowExW.argtypes = [
    wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID
]

class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [("usUsagePage", wintypes.USHORT), ("usUsage", wintypes.USHORT),
                ("dwFlags", wintypes.DWORD), ("hwndTarget", wintypes.HWND)]

class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [("dwType", wintypes.DWORD), ("dwSize", wintypes.DWORD),
                ("hDevice", wintypes.HANDLE), ("wParam", wintypes.WPARAM)]

class RAWMOUSE(ctypes.Structure):
    _fields_ = [("usFlags", wintypes.USHORT), ("usButtonFlags", wintypes.USHORT),
                ("usButtonData", wintypes.USHORT), ("ulRawButtons", wintypes.ULONG),
                ("lLastX", wintypes.LONG), ("lLastY", wintypes.LONG),
                ("ulExtraInformation", wintypes.ULONG)]

class RAWINPUT(ctypes.Structure):
    class _DATA(ctypes.Union):
        _fields_ = [("mouse", RAWMOUSE), ("keyboard", wintypes.BYTE * 24), ("hid", wintypes.BYTE * 24)]
    _fields_ = [("header", RAWINPUTHEADER), ("data", _DATA)]
    _anonymous_ = ("data",)

class WNDCLASSW(ctypes.Structure):
    _fields_ = [("style", wintypes.UINT),
                ("lpfnWndProc", ctypes.CFUNCTYPE(wintypes.LONG, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)),
                ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int),
                ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HANDLE),
                ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HANDLE),
                ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR)]

def get_vrchat_window_bbox():
    global vrchat_hwnd
    hwnd = win32gui.FindWindow(None, "VRChat")
    if hwnd == 0:
        print("[ERRO] Janela do VRChat não encontrada!")
        return None
    vrchat_hwnd = hwnd
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    win32gui.SetForegroundWindow(hwnd)
    return win32gui.GetWindowRect(hwnd)

def on_press(key):
    try:
        if hasattr(key, 'char') and key.char:
            c = key.char.lower()
            pressed_keys.add(c)
            if c == 'p':
                is_paused[0] = not is_paused[0]
                print("[INFO] Gravação pausada." if is_paused[0] else "[INFO] Gravação retomada.")
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

def save_label(image_name, keys, dx, dy, timestamp):
    with open(LABEL_FILE, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([image_name, "+".join(keys), dx, dy, f"{timestamp:.3f}"])

def raw_input_thread():
    global mouse_dx, mouse_dy
    WNDPROC = ctypes.CFUNCTYPE(wintypes.LONG, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

    @WNDPROC
    def wnd_proc(hwnd, msg, wparam, lparam):
        global mouse_dx, mouse_dy
        if msg == WM_INPUT:
            dwSize = wintypes.UINT()
            user32.GetRawInputData(lparam, 0x10000003, None, ctypes.byref(dwSize), ctypes.sizeof(RAWINPUTHEADER))
            raw_buffer = ctypes.create_string_buffer(dwSize.value)
            if user32.GetRawInputData(lparam, 0x10000003, raw_buffer, ctypes.byref(dwSize), ctypes.sizeof(RAWINPUTHEADER)) == dwSize.value:
                raw = ctypes.cast(raw_buffer, ctypes.POINTER(RAWINPUT)).contents
                if raw.header.dwType == 0:
                    mouse_dx += raw.mouse.lLastX
                    mouse_dy += raw.mouse.lLastY
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    wnd_class = WNDCLASSW()
    wnd_class.lpfnWndProc = wnd_proc
    h_instance = kernel32.GetModuleHandleW(None)
    wnd_class.hInstance = h_instance
    wnd_class.lpszClassName = "RawInputWindow"
    user32.RegisterClassW(ctypes.byref(wnd_class))

    hwnd = user32.CreateWindowExW(0, wnd_class.lpszClassName, "RawInput", 0, 0, 0, 0, 0, None, None, h_instance, None)
    rid = RAWINPUTDEVICE()
    rid.usUsagePage = 0x01
    rid.usUsage = 0x02
    rid.dwFlags = 0x00000100
    rid.hwndTarget = hwnd
    user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(rid))

    msg = wintypes.MSG()
    while recording[0]:
        if user32.GetMessageW(ctypes.byref(msg), hwnd, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        time.sleep(0.001)
    user32.DestroyWindow(hwnd)

def image_saver_worker():
    while recording[0] or not image_queue.empty():
        try:
            img, filepath = image_queue.get(timeout=0.1)
            if IMAGE_FORMAT.lower() == "jpeg":
                img = img.convert("RGB")  # JPEG não suporta transparência
                img.save(filepath, format="JPEG", quality=JPEG_QUALITY)
            else:
                img.save(filepath, format="PNG")
            image_queue.task_done()
        except Empty:
            continue
    print("[INFO] Image saver worker finalizado.")

def label_saver_worker():
    while recording[0] or not label_queue.empty():
        try:
            item = label_queue.get(timeout=0.1)
            save_label(*item)
            label_queue.task_done()
        except Empty:
            continue
    print("[INFO] Label saver worker finalizado.")

def record():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(LABEL_FILE):
        with open(LABEL_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["image", "keys", "mouse_dx", "mouse_dy", "timestamp"])

    rect = get_vrchat_window_bbox()
    if not rect:
        print("[ERRO] Janela do VRChat não encontrada.")
        return

    for t in [image_saver_worker, label_saver_worker]:
        for _ in range(NUM_WORKER_THREADS):
            threading.Thread(target=t, daemon=True).start()

    global mouse_dx, mouse_dy
    start_time_global = time.time()

    while recording[0]:
        if is_paused[0]:
            time.sleep(0.1)
            continue

        start_time = time.time()
        now = time.time()
        timestamp = now - start_time_global

        img = ImageGrab.grab(bbox=rect).resize((224, 224))
        filename = f"{uuid.uuid4()}.{IMAGE_FORMAT}"
        filepath = os.path.join(DATA_DIR, filename)

        image_queue.put((img, filepath))
        label_queue.put((filename, pressed_keys.copy(), mouse_dx, mouse_dy, timestamp))

        mouse_dx = 0
        mouse_dy = 0

        elapsed = time.time() - start_time
        if elapsed < INTERVAL:
            time.sleep(INTERVAL - elapsed)

    print("[INFO] Finalizando gravação, aguardando filas...")
    image_queue.join()
    label_queue.join()

def main():
    print("Iniciando gravação. Pressione ESC para parar, P para pausar/resumir.")
    keyboard.Listener(on_press=on_press, on_release=on_release).start()
    threading.Thread(target=raw_input_thread, daemon=True).start()
    record()
    print("Gravação finalizada.")

if __name__ == "__main__":
    main()
