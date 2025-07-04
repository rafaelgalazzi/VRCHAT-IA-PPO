import torch
import numpy as np
import time
import json
import os
from torchvision import transforms
from PIL import Image

from agents.imitation_agent import ImitationAgent
from utils.input_controller import key_down, key_up, move_mouse_relative
from utils.screen_utils import capture_vrchat_frame

# ---- Configurações ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
KEYS = ["w", "s", "shift", "space", "a", "d"]
SEQ_LEN = 6  # número de frames empilhados
FRAME_DELAY = 0.05
INPUT_CHANNELS = 3 * SEQ_LEN

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
basic_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---- Modelos ----
keyboard_model = ImitationAgent(output_dim=6, mode="keyboard", input_channels=INPUT_CHANNELS).to(DEVICE)
mouse_model = ImitationAgent(output_dim=2, mode="mouse", input_channels=INPUT_CHANNELS).to(DEVICE)

keyboard_model.load_state_dict(torch.load("imitation_keyboard_latest.pth", map_location=DEVICE))
mouse_model.load_state_dict(torch.load("imitation_mouse_latest.pth", map_location=DEVICE))
keyboard_model.eval()
mouse_model.eval()

# ---- Frame buffer ----
frame_buffer = []

print("[INFO] Modelos carregados. Iniciando inferência com contexto temporal... Pressione Ctrl+C para parar.")

try:
    while True:
        img = capture_vrchat_frame()
        if img is None:
            print("[AVISO] Imagem não capturada.")
            time.sleep(FRAME_DELAY)
            continue

        tensor = basic_transform(img)
        frame_buffer.append(tensor)

        if len(frame_buffer) < SEQ_LEN:
            time.sleep(FRAME_DELAY)
            continue
        elif len(frame_buffer) > SEQ_LEN:
            frame_buffer.pop(0)

        stacked = torch.cat(frame_buffer, dim=0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            key_probs = keyboard_model(stacked).squeeze().cpu().numpy()
            mouse_movement = mouse_model(stacked).squeeze().cpu().numpy()

        # Aplicar teclas
        for i, key in enumerate(KEYS):
            if key_probs[i] > 0.5:
                key_down(key)
            else:
                key_up(key)

        # Movimento do mouse
        dx = int(mouse_movement[0] * mouse_norm["max_dx"])
        dy = int(mouse_movement[1] * mouse_norm["max_dy"])
        move_mouse_relative(dx, dy)

        time.sleep(FRAME_DELAY)

except KeyboardInterrupt:
    print("\n[INFO] Interrompido pelo usuário.")
    for key in KEYS:
        key_up(key)
