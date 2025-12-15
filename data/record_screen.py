import os
import csv
import time
import queue
from datetime import datetime
import numpy as np
import pyautogui
from pynput import mouse, keyboard

# Local imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.screen_utils import capture_window_frame
from utils.input_controller import MOUSE_BUTTONS

# Configuration
IMAGE_DIR = "data/images"
LABEL_FILE = "data/labels.csv"
MAX_RECORDING_DURATION = 3600  # Maximum recording duration in seconds
FPS = 20  # Frames per second
FRAME_DELAY = 1.0 / FPS

# Create directories
os.makedirs(IMAGE_DIR, exist_ok=True)

class InputRecorder:
    def __init__(self):
        self.start_time = time.time()
        self.current_keys = set()
        self.current_mouse_buttons = set()
        self.events_queue = queue.Queue()
        self.recording = False
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.quit_pressed = False  # Track if quit key was pressed

        # Mouse listener
        self.mouse_listener = mouse.Listener(
            on_move=self.on_mouse_move,
            on_click=self.on_mouse_click,
            on_scroll=self.on_mouse_scroll
        )

        # Keyboard listener
        self.keyboard_listener = keyboard.Listener(
            on_press=self.on_key_press,
            on_release=self.on_key_release
        )

    def on_key_press(self, key):
        try:
            # Handle regular character keys
            if hasattr(key, 'char') and key.char is not None:
                self.current_keys.add(key.char.upper())
            elif key == keyboard.Key.esc:
                self.quit_pressed = True
            elif key == keyboard.Key.space:
                self.current_keys.add('SPACE')
            elif key == keyboard.Key.enter:
                self.current_keys.add('ENTER')
            elif key == keyboard.Key.tab:
                self.current_keys.add('TAB')
            elif key == keyboard.Key.backspace:
                self.current_keys.add('BACKSPACE')
            elif key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                self.current_keys.add('CTRL')
            elif key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
                self.current_keys.add('ALT')
            elif key == keyboard.Key.shift_l or key == keyboard.Key.shift_r:
                self.current_keys.add('SHIFT')
            elif str(key).startswith('Key.'):
                # Handle other special keys like Key.f1, Key.up, etc.
                special_key = str(key).replace('Key.', '').upper()
                self.current_keys.add(special_key)
        except Exception as e:
            print(f"[DEBUG] Error in on_key_press: {e}")

    def on_key_release(self, key):
        try:
            # Handle regular character keys
            if hasattr(key, 'char') and key.char is not None:
                self.current_keys.discard(key.char.upper())
            elif key == keyboard.Key.esc:
                self.quit_pressed = False  # Reset quit state when ESC is released
            elif key == keyboard.Key.space:
                self.current_keys.discard('SPACE')
            elif key == keyboard.Key.enter:
                self.current_keys.discard('ENTER')
            elif key == keyboard.Key.tab:
                self.current_keys.discard('TAB')
            elif key == keyboard.Key.backspace:
                self.current_keys.discard('BACKSPACE')
            elif key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
                self.current_keys.discard('CTRL')
            elif key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
                self.current_keys.discard('ALT')
            elif key == keyboard.Key.shift_l or key == keyboard.Key.shift_r:
                self.current_keys.discard('SHIFT')
            elif str(key).startswith('Key.'):
                # Handle other special keys like Key.f1, Key.up, etc.
                special_key = str(key).replace('Key.', '').upper()
                self.current_keys.discard(special_key)
        except Exception as e:
            print(f"[DEBUG] Error in on_key_release: {e}")

    def on_mouse_move(self, x, y):
        # Record mouse movement if needed
        pass

    def on_mouse_click(self, x, y, button, pressed):
        # Convert button to string and extract the actual button name
        button_str = str(button)
        # Format is "Button.left", "Button.right", etc.
        if 'left' in button_str:
            button_name = 'L'
        elif 'right' in button_str:
            button_name = 'R'
        elif 'middle' in button_str:
            button_name = 'M'
        else:
            # Extract button name from string representation
            button_name = str(button).replace('Button.', '').upper()

        if pressed:
            self.current_mouse_buttons.add(button_name)
        else:
            self.current_mouse_buttons.discard(button_name)

    def on_mouse_scroll(self, x, y, dx, dy):
        # Handle scroll if needed
        pass

    def start_recording(self):
        self.recording = True
        self.start_time = time.time()
        self.mouse_listener.start()
        self.keyboard_listener.start()

        print("[INFO] Recording started. Press ESC to quit.")

        # Initialize CSV file - truncate if it exists to start fresh
        fieldnames = ['timestamp', 'session_id', 'image', 'keys', 'mouse_dx', 'mouse_dy', 'mouse_buttons']
        with open(LABEL_FILE, 'w', newline='') as csvfile:  # Use 'w' to overwrite and start fresh
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

        # Main recording loop
        last_mouse_pos = pyautogui.position()
        frame_count = 0

        try:
            while self.recording:
                current_time = time.time()

                # Check for quit condition - use the flag we set in the keyboard handler
                if self.quit_pressed:
                    print("[INFO] Quit key (ESC) detected. Stopping recording...")
                    break

                # Check duration limit
                if current_time - self.start_time > MAX_RECORDING_DURATION:
                    print("[INFO] Maximum recording duration reached.")
                    break

                # Capture frame
                img = capture_window_frame()
                if img is None:
                    print("[WARNING] Could not capture frame, skipping...")
                    time.sleep(FRAME_DELAY)
                    continue

                # Save image
                timestamp = int(current_time * 1000)  # milliseconds
                image_filename = f"frame_{self.session_id}_{frame_count:06d}.png"
                image_path = os.path.join(IMAGE_DIR, image_filename)
                img.save(image_path)

                # Calculate mouse movement since last frame
                current_mouse_pos = pyautogui.position()
                dx = current_mouse_pos.x - last_mouse_pos.x
                dy = current_mouse_pos.y - last_mouse_pos.y
                last_mouse_pos = current_mouse_pos

                # Write to CSV
                with open(LABEL_FILE, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow({
                        'timestamp': timestamp,
                        'session_id': self.session_id,
                        'image': image_filename,
                        'keys': '+'.join(sorted(self.current_keys)),
                        'mouse_dx': dx,
                        'mouse_dy': dy,
                        'mouse_buttons': '+'.join(sorted(self.current_mouse_buttons))
                    })

                frame_count += 1
                print(f"[INFO] Frame {frame_count} recorded - Keys: {sorted(self.current_keys)}, Buttons: {sorted(self.current_mouse_buttons)}", end='\r')

                time.sleep(FRAME_DELAY)

        except KeyboardInterrupt:
            print("\n[INFO] Recording interrupted by user.")

        finally:
            self.stop_recording()

    def stop_recording(self):
        self.recording = False
        if self.mouse_listener.is_alive():
            self.mouse_listener.stop()
        if self.keyboard_listener.is_alive():
            self.keyboard_listener.stop()
        print(f"\n[INFO] Recording stopped. Total frames: {int((time.time() - self.start_time) * FPS)}")

def main():
    recorder = InputRecorder()
    try:
        recorder.start_recording()
    except Exception as e:
        print(f"[ERROR] Recording failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()