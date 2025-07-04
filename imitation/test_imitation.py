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
from agents.imitation_agent import ImitationAgent
from utils.screen_utils import capture_vrchat_frame
from utils.input_controller import key_down, key_up, KEYS, move_mouse_relative

# ---- Configuração ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KEY_LIST = list(KEYS.keys())
NORM_PATH = "data/mouse_normalization.json"
SEQ_LEN = 6  # número de frames empilhados
FRAME_DELAY = 0.05  # tempo entre capturas (50ms ~ 20fps)

# ---- Normalização do mouse ----
with open(NORM_PATH, "r") as f:
    mouse_norm = json.load(f)
max_dx = mouse_norm["max_dx"]
max_dy = mouse_norm["max_dy"]

# ---- Transformação de imagem (por frame) ----
basic_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---- Modelos ----
input_channels = 3 * SEQ_LEN
keyboard_model = ImitationAgent(output_dim=len(KEY_LIST), mode="keyboard", input_channels=input_channels).to(DEVICE)
mouse_model = ImitationAgent(output_dim=2, mode="mouse", input_channels=input_channels).to(DEVICE)

keyboard_model.load_state_dict(torch.load("imitation_keyboard_latest.pth", map_location=DEVICE))
mouse_model.load_state_dict(torch.load("imitation_mouse_latest.pth", map_location=DEVICE))
keyboard_model.eval()
mouse_model.eval()

# ---- Buffer de frames ----
frame_buffer = []

print(f"[INFO] Rodando teclado + mouse com contexto temporal ({SEQ_LEN} frames) - Pressione ESC para sair.")
prev_keys = np.zeros(len(KEY_LIST))

try:
    while True:
        if keyboard.is_pressed('esc'):
            break

        img = capture_vrchat_frame()
        if img is None:
            continue

        # Processa imagem e atualiza buffer
        tensor = basic_transform(img)
        frame_buffer.append(tensor)
        if len(frame_buffer) < SEQ_LEN:
            time.sleep(FRAME_DELAY)
            continue
        elif len(frame_buffer) > SEQ_LEN:
            frame_buffer.pop(0)

        # Empilha as imagens
        stacked = torch.cat(frame_buffer, dim=0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            key_output = keyboard_model(stacked).cpu().numpy().squeeze()
            mouse_output = mouse_model(stacked).cpu().numpy().squeeze()

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

        time.sleep(FRAME_DELAY)

finally:
    for key in KEY_LIST:
        key_up(key)
    print("[INFO] Execução finalizada.")
