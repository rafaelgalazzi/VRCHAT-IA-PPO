# Mouse and keyboard control
import ctypes

KEYS = {
    "w": 0x57, "s": 0x53, "shift": 0x10, "space": 0x20,
    "a": 0x41, "d": 0x44,
}

SendInput = ctypes.windll.user32.SendInput
PUL = ctypes.POINTER(ctypes.c_ulong)

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL)
    ]

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL)
    ]

class INPUT_I(ctypes.Union):
    _fields_ = [
        ("ki", KEYBDINPUT),
        ("mi", MOUSEINPUT)
    ]

class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("ii", INPUT_I)
    ]

def send_key(hexKeyCode, down=True):
    scan_code = ctypes.windll.user32.MapVirtualKeyW(hexKeyCode, 0)
    flags = 0x0008  # KEYEVENTF_SCANCODE
    if not down:
        flags |= 0x0002  # KEYEVENTF_KEYUP
    ki = KEYBDINPUT(0, scan_code, flags, 0, ctypes.pointer(ctypes.c_ulong(0)))
    x = INPUT(1, INPUT_I(ki=ki))
    return SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def key_down(key):
    if key in KEYS:
        send_key(KEYS[key], True)

def key_up(key):
    if key in KEYS:
        send_key(KEYS[key], False)

def move_mouse_relative(dx, dy):
    mi = MOUSEINPUT(int(dx), int(dy), 0, 0x0001, 0, ctypes.pointer(ctypes.c_ulong(0)))
    x = INPUT(0, INPUT_I(mi=mi))
    return SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
