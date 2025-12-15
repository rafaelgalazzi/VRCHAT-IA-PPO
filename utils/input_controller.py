# Mouse and keyboard control via Arduino serial protocol
import serial
import time

HOLD_MS = 80  # Key and mouse press hold duration in milliseconds
BAUD_RATE = 115200

# Key mapping - all supported keys by Arduino
KEY_MAP = {
    # Alphabetic keys (A-Z)
    "A": "A", "B": "B", "C": "C", "D": "D", "E": "E",
    "F": "F", "G": "G", "H": "H", "I": "I", "J": "J",
    "K": "K", "L": "L", "M": "M", "N": "N", "O": "O",
    "P": "P", "Q": "Q", "R": "R", "S": "S", "T": "T",
    "U": "U", "V": "V", "W": "W", "X": "X", "Y": "Y",
    "Z": "Z",
    
    # Numeric keys (0-9)
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4",
    "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
    
    # Function keys (F1-F12)
    "F1": "F1", "F2": "F2", "F3": "F3", "F4": "F4",
    "F5": "F5", "F6": "F6", "F7": "F7", "F8": "F8",
    "F9": "F9", "F10": "F10", "F11": "F11", "F12": "F12",
    
    # Special keys
    "ENTER": "ENTER", "RETURN": "RETURN",
    "ESC": "ESC", "ESCAPE": "ESCAPE",
    "TAB": "TAB",
    "SPACE": "SPACE",
    "BACKSPACE": "BACKSPACE",
    
    # Arrow keys
    "UP": "UP",
    "DOWN": "DOWN",
    "LEFT": "LEFT",
    "RIGHT": "RIGHT",
    
    # Modifier keys
    "CTRL": "CTRL", "CONTROL": "CONTROL",
    "SHIFT": "SHIFT",
    "ALT": "ALT",
    "GUI": "GUI", "WIN": "WIN", "CMD": "CMD",
    
    # Left/Right variants
    "CTRL_L": "CTRL_L", "CTRL_R": "CTRL_R",
    "SHIFT_L": "SHIFT_L", "SHIFT_R": "SHIFT_R",
    "ALT_L": "ALT_L", "ALT_R": "ALT_R",
    "GUI_L": "GUI_L", "GUI_R": "GUI_R",
}

# Mouse button mapping
MOUSE_MAP = {
    "L": "L",      # Left button
    "R": "R",      # Right button
    "M": "M",      # Middle button
}

# Common key sets for compatibility with training data
TRAINING_KEYS = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"]  # Standard game controls used in training

# Mouse buttons for compatibility
MOUSE_BUTTONS = ["L", "R", "M"]  # Left, Right, Middle buttons

# Global serial connection
arduino = None

def init_serial(port='COM3', baudrate=BAUD_RATE, timeout=1):
    """Initialize serial connection to Arduino."""
    global arduino
    try:
        arduino = serial.Serial(port, baudrate, timeout=timeout)
        time.sleep(1.5)  # Wait for Arduino to be ready
        # Wait for READY response
        response = arduino.readline().decode('utf-8').strip()
        if response == "READY":
            print("Arduino connected and ready!")
            return True
        else:
            print(f"Unexpected response: {response}")
            return False
    except serial.SerialException as e:
        print(f"Error connecting to Arduino: {e}")
        return False

def close_serial():
    """Close serial connection."""
    global arduino
    if arduino and arduino.is_open:
        arduino.close()

def send_command(command):
    """Send a command to Arduino and get response."""
    global arduino
    if arduino is None or not arduino.is_open:
        print("Arduino not connected")
        return False
    
    try:
        # Send command
        arduino.write((command + '\n').encode('utf-8'))
        
        # Read response
        response = arduino.readline().decode('utf-8').strip()
        if response == "OK":
            return True
        else:
            print(f"Arduino error: {response}")
            return False
    except Exception as e:
        print(f"Error sending command: {e}")
        return False

def type_string(text):
    """Type a string of text."""
    send_command(f"TYPE {text}")

def key_press(key_name):
    """Press and release a key with HOLD_MS delay."""
    send_command(f"PRESS {key_name}")

def key_down(key_name):
    """Press a key down."""
    send_command(f"DOWN {key_name}")

def key_up(key_name):
    """Release a key."""
    send_command(f"UP {key_name}")

def mouse_click(button):
    """Click a mouse button (L/R/M)."""
    send_command(f"CLICK {button.upper()}")

def move_mouse_relative(dx, dy):
    """Move mouse by relative amount (dx, dy)."""
    send_command(f"MOVE {int(dx)} {int(dy)}")

def mouse_wheel(w):
    """Scroll mouse wheel."""
    send_command(f"WHEEL {int(w)}")

def enable():
    """Enable Arduino input."""
    send_command("ENABLE")

def disable():
    """Disable Arduino input."""
    send_command("DISABLE")

def ping():
    """Ping Arduino to test connection."""
    send_command("PING")

def mouse_down(button):
    """Press mouse button down."""
    send_command(f"DOWN_MOUSE {button.upper()}")

def mouse_up(button):
    """Release mouse button."""
    send_command(f"UP_MOUSE {button.upper()}")
