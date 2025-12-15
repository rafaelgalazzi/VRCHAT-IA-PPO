import torch
import numpy as np
import time
import keyboard
import json
import sys
import os
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agents.imitation_agent import ImitationAgentLSTM
from utils.screen_utils import capture_window_frame
from utils.input_controller import key_down, key_up, move_mouse_relative, mouse_click, mouse_down, mouse_up, TRAINING_KEYS, MOUSE_BUTTONS

# ---- Configuração ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KEY_LIST = TRAINING_KEYS  # Specific keys used in training
MOUSE_BUTTONS_LIST = MOUSE_BUTTONS  # Mouse buttons used in training
NORM_PATH = "data/mouse_normalization.json"
SEQ_LEN = 6
FRAME_DELAY = 0.05

# ---- Normalização do mouse ----
with open(NORM_PATH, "r") as f:
    mouse_norm = json.load(f)
max_dx = mouse_norm["max_dx"]
max_dy = mouse_norm["max_dy"]

# ---- Transformação de imagem ----
basic_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---- Modelos LSTM ----
keyboard_model = ImitationAgentLSTM(output_dim=len(KEY_LIST), mode="keyboard").to(DEVICE)
mouse_model = ImitationAgentLSTM(output_dim=2, mode="mouse").to(DEVICE)
mouse_click_model = ImitationAgentLSTM(output_dim=len(MOUSE_BUTTONS_LIST), mode="mouse_click").to(DEVICE)

keyboard_model.load_state_dict(torch.load("imitation_keyboard_latest.pth", map_location=DEVICE))
mouse_model.load_state_dict(torch.load("imitation_mouse_latest.pth", map_location=DEVICE))
mouse_click_model.load_state_dict(torch.load("imitation_mouse_click_latest.pth", map_location=DEVICE))
keyboard_model.eval()
mouse_model.eval()
mouse_click_model.eval()

# ---- Buffer de frames ----
frame_buffer = []

print(f"[INFO] Rodando teclado + mouse + mouse clicks com contexto temporal ({SEQ_LEN} frames via LSTM) - Pressione ESC para sair.")
prev_keys = np.zeros(len(KEY_LIST))
prev_mouse_buttons = np.zeros(len(MOUSE_BUTTONS_LIST))

try:
    while True:
        if keyboard.is_pressed('esc'):
            break

        img = capture_window_frame()
        if img is None:
            continue

        tensor = basic_transform(img)
        frame_buffer.append(tensor)

        if len(frame_buffer) < SEQ_LEN:
            time.sleep(FRAME_DELAY)
            continue
        elif len(frame_buffer) > SEQ_LEN:
            frame_buffer.pop(0)

        # Constrói o batch [1, T, 3, H, W]
        sequence = torch.stack(frame_buffer, dim=0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            key_output = keyboard_model(sequence).cpu().numpy().squeeze()
            mouse_output = mouse_model(sequence).cpu().numpy().squeeze()
            mouse_click_output = mouse_click_model(sequence).cpu().numpy().squeeze()

        # ---- Teclado ----
        key_states = (key_output > 0.5).astype(int)
        for i, key in enumerate(KEY_LIST):
            if key_states[i] and not prev_keys[i]:
                key_down(key)
            elif not key_states[i] and prev_keys[i]:
                key_up(key)
        prev_keys = key_states

        # ---- Mouse ----
        dx = int(mouse_output[0] * max_dx)
        dy = int(mouse_output[1] * max_dy)
        move_mouse_relative(dx, dy)

        # ---- Mouse Buttons ----
        mouse_button_states = (mouse_click_output > 0.5).astype(int)
        for i, button in enumerate(MOUSE_BUTTONS_LIST):
            if mouse_button_states[i] and not prev_mouse_buttons[i]:
                mouse_down(button)  # Press mouse button down
            elif not mouse_button_states[i] and prev_mouse_buttons[i]:
                mouse_up(button)    # Release mouse button
        prev_mouse_buttons = mouse_button_states

        time.sleep(FRAME_DELAY)

finally:
    for key in KEY_LIST:
        key_up(key)
    for button in MOUSE_BUTTONS_LIST:
        mouse_up(button)
    print("[INFO] Execução finalizada.")
