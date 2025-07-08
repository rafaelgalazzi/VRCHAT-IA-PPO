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
from utils.input_controller import key_down, key_up, move_mouse_relative
from utils.screen_utils import capture_vrchat_frame

# ---- Configurações ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KEYS = ["w", "s", "shift", "space", "a", "d"]
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
keyboard_model = ImitationAgentLSTM(output_dim=6, mode="keyboard").to(DEVICE)
mouse_model = ImitationAgentLSTM(output_dim=2, mode="mouse").to(DEVICE)

keyboard_model.load_state_dict(torch.load("imitation_keyboard_latest.pth", map_location=DEVICE))
mouse_model.load_state_dict(torch.load("imitation_mouse_latest.pth", map_location=DEVICE))
keyboard_model.eval()
mouse_model.eval()

# ---- Buffer de frames ----
frame_buffer = deque(maxlen=SEQ_LEN)

print("[INFO] Starting inference... Press ESC to stop.")

try:
    while True:
        if keyboard.is_pressed("esc"):
            print("\n[INFO] Interrupted by ESC. Releasing keys...")
            break

        img = capture_vrchat_frame()
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

        for i, key in enumerate(KEYS):
            if key_probs[i] > 0.5:
                key_down(key)
            else:
                key_up(key)

        dx = int(mouse_movement[0] * mouse_norm["max_dx"])
        dy = int(mouse_movement[1] * mouse_norm["max_dy"])
        move_mouse_relative(dx, dy)

        time.sleep(FRAME_DELAY)

except Exception as e:
    print(f"[ERRO] {e}")

finally:
    for key in KEYS:
        key_up(key)
    print("[INFO] Releasing keys... Finished.")
