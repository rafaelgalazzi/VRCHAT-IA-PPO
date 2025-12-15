import psutil
import win32gui
import win32process

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
    return result

hwnds = find_window_by_process("pbRO.exe")


for hwnd in hwnds:
    print(hwnd, win32gui.GetWindowText(hwnd))
print(hwnds)