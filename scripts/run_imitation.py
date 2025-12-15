import torch
import numpy as np
import time
import json
import os
from collections import deque
from torchvision import transforms
from PIL import Image
import keyboard  # <- Adicionado para ESC

from agents.imitation_agent import ImitationAgentLSTM
from utils.input_controller import key_down, key_up, move_mouse_relative, mouse_click, mouse_down, mouse_up, TRAINING_KEYS, MOUSE_BUTTONS
from utils.screen_utils import capture_window_frame

# ---- Configurações ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KEYS = TRAINING_KEYS  # Specific keys used in training
MOUSE_BUTTONS_LIST = MOUSE_BUTTONS  # Mouse buttons used in training
SEQ_LEN = 6
FRAME_DELAY = 0.05

# ---- Normalização do mouse ----
NORMALIZATION_PATH = "data/mouse_normalization.json"
mouse_norm = {"max_dx": 1.0, "max_dy": 1.0}
if os.path.exists(NORMALIZATION_PATH):
    with open(NORMALIZATION_PATH, "r") as f:
        mouse_norm = json.load(f)
    print(f"[INFO] Normalização de mouse carregada: {mouse_norm}")
else:
    print(f"[AVISO] Arquivo de normalização não encontrado: {NORMALIZATION_PATH}")

# ---- Transformação da imagem ----
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---- Carregamento dos modelos ----
keyboard_model = ImitationAgentLSTM(output_dim=len(TRAINING_KEYS), mode="keyboard").to(DEVICE)
mouse_model = ImitationAgentLSTM(output_dim=2, mode="mouse").to(DEVICE)
mouse_click_model = ImitationAgentLSTM(output_dim=len(MOUSE_BUTTONS_LIST), mode="mouse_click").to(DEVICE)

keyboard_model.load_state_dict(torch.load("imitation_keyboard_latest.pth", map_location=DEVICE))
mouse_model.load_state_dict(torch.load("imitation_mouse_latest.pth", map_location=DEVICE))
mouse_click_model.load_state_dict(torch.load("imitation_mouse_click_latest.pth", map_location=DEVICE))
keyboard_model.eval()
mouse_model.eval()
mouse_click_model.eval()

# ---- Buffer de frames ----
frame_buffer = deque(maxlen=SEQ_LEN)

print("[INFO] Starting inference with keyboard, mouse movement, and mouse clicks... Press ESC to stop.")

# Initialize previous state variables
prev_key_states = np.zeros(len(KEYS))
prev_mouse_button_states = np.zeros(len(MOUSE_BUTTONS_LIST))

try:
    while True:
        if keyboard.is_pressed("esc"):
            print("\n[INFO] Interrupted by ESC. Releasing keys...")
            break

        img = capture_window_frame()
        if img is None:
            print("[WARNING] Image not captured.")
            time.sleep(FRAME_DELAY)
            continue

        tensor = transform(img)
        frame_buffer.append(tensor)

        if len(frame_buffer) < SEQ_LEN:
            time.sleep(FRAME_DELAY)
            continue

        # [1, T, 3, H, W]
        sequence = torch.stack(list(frame_buffer), dim=0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            key_probs = keyboard_model(sequence).squeeze().cpu().numpy()
            mouse_movement = mouse_model(sequence).squeeze().cpu().numpy()
            mouse_click_probs = mouse_click_model(sequence).squeeze().cpu().numpy()

        # Handle keyboard inputs
        key_states = (key_probs > 0.5).astype(int)
        for i, key in enumerate(KEYS):
            if key_states[i] and not prev_key_states[i]:
                key_down(key)
            elif not key_states[i] and prev_key_states[i]:
                key_up(key)
        prev_key_states = key_states

        # Handle mouse movement
        dx = int(mouse_movement[0] * mouse_norm["max_dx"])
        dy = int(mouse_movement[1] * mouse_norm["max_dy"])
        move_mouse_relative(dx, dy)

        # Handle mouse button clicks
        mouse_button_states = (mouse_click_probs > 0.5).astype(int)
        for i, button in enumerate(MOUSE_BUTTONS_LIST):
            if mouse_button_states[i] and not prev_mouse_button_states[i]:
                mouse_down(button)  # Press mouse button down
            elif not mouse_button_states[i] and prev_mouse_button_states[i]:
                mouse_up(button)    # Release mouse button
        prev_mouse_button_states = mouse_button_states

        time.sleep(FRAME_DELAY)

except Exception as e:
    print(f"[ERRO] {e}")

finally:
    for key in KEYS:
        key_up(key)
    for button in MOUSE_BUTTONS_LIST:
        mouse_up(button)
    print("[INFO] Releasing keys and mouse buttons... Finished.")
