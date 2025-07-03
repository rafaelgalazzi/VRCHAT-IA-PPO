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

# ---- Normalização do mouse ----
with open(NORM_PATH, "r") as f:
    mouse_norm = json.load(f)
max_dx = mouse_norm["max_dx"]
max_dy = mouse_norm["max_dy"]

# ---- Carrega modelos ----
keyboard_model = ImitationAgent(output_dim=len(KEY_LIST), mode="keyboard").to(DEVICE)
mouse_model = ImitationAgent(output_dim=2, mode="mouse").to(DEVICE)
keyboard_model.load_state_dict(torch.load("imitation_keyboard_latest.pth", map_location=DEVICE))
mouse_model.load_state_dict(torch.load("imitation_mouse_latest.pth", map_location=DEVICE))
keyboard_model.eval()
mouse_model.eval()

# ---- Transformação de imagem ----
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

print(f"[INFO] Rodando teclado + mouse - Pressione ESC para sair.")
prev_keys = np.zeros(len(KEY_LIST))

try:
    while True:
        if keyboard.is_pressed('esc'):
            break

        img = capture_vrchat_frame()
        if img is None:
            continue

        with torch.no_grad():
            tensor = transform(img).unsqueeze(0).to(DEVICE)
            key_output = keyboard_model(tensor).cpu().numpy().squeeze()
            mouse_output = mouse_model(tensor).cpu().numpy().squeeze()

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

        time.sleep(0.05)

finally:
    for key in KEY_LIST:
        key_up(key)
    print("[INFO] Execução finalizada.")
