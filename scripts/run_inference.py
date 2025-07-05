import time
import os
import json
import sys
from collections import deque
from PIL import ImageGrab
import numpy as np
import torch
import keyboard  # <- Adicionado para ESC
from torchvision import transforms

# Importações internas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.input_controller import key_down, key_up, move_mouse_relative
from agents.imitation_agent import ImitationAgentLSTM
from agents.ppo_agent import get_vrchat_window_bbox

# === CONFIGURAÇÕES ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
FPS = 10
INTERVAL = 1.0 / FPS
KEY_NAMES = ["w", "s", "shift", "space", "a", "d"]
SEQ_LEN = 6  # mesmo usado no treino

# === NORMALIZAÇÃO DO MOUSE ===
NORMALIZATION_PATH = "data/mouse_normalization.json"
mouse_norm = {"max_dx": 1.0, "max_dy": 1.0}
if os.path.exists(NORMALIZATION_PATH):
    with open(NORMALIZATION_PATH, "r") as f:
        mouse_norm = json.load(f)
    print(f"[INFO] Normalização de mouse carregada: {mouse_norm}")
else:
    print(f"[AVISO] Arquivo de normalização não encontrado. Usando padrão.")

# === TRANSFORMAÇÃO DE IMAGEM ===
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === CARREGAMENTO DOS MODELOS ===
keyboard_model = ImitationAgentLSTM(output_dim=6, mode="keyboard").to(DEVICE)
mouse_model = ImitationAgentLSTM(output_dim=2, mode="mouse").to(DEVICE)

keyboard_model.load_state_dict(torch.load("imitation_keyboard_latest.pth", map_location=DEVICE))
mouse_model.load_state_dict(torch.load("imitation_mouse_latest.pth", map_location=DEVICE))
keyboard_model.eval()
mouse_model.eval()

# === VERIFICAÇÃO DE JANELA ===
bbox = get_vrchat_window_bbox()
if not bbox:
    sys.exit(1)

frame_stack = deque(maxlen=SEQ_LEN)
prev_keys = np.zeros(len(KEY_NAMES))

print("[INFO] Iniciando inferência... Pressione ESC para encerrar.")

try:
    while True:
        if keyboard.is_pressed("esc"):
            print("\n[INFO] Interrompido via ESC. Liberando teclas...")
            break

        start = time.time()

        # Captura de tela e empilhamento
        img = ImageGrab.grab(bbox=bbox).resize((IMAGE_SIZE, IMAGE_SIZE))
        frame_tensor = transform(img)  # [3, H, W]
        frame_stack.append(frame_tensor)

        # Preenche se necessário
        while len(frame_stack) < SEQ_LEN:
            frame_stack.append(frame_tensor.clone())

        # Cria sequência [1, T, 3, H, W]
        seq_tensor = torch.stack(list(frame_stack)).unsqueeze(0).to(DEVICE)

        # Inferência
        with torch.no_grad():
            key_output = keyboard_model(seq_tensor).cpu().numpy()[0]  # [6]
            mouse_output = mouse_model(seq_tensor).cpu().numpy()[0]   # [2]

        # Ações de teclado
        key_pressed = (key_output > 0.5).astype(int)
        for i, key_name in enumerate(KEY_NAMES):
            if key_pressed[i] and not prev_keys[i]:
                key_down(key_name)
            elif not key_pressed[i] and prev_keys[i]:
                key_up(key_name)
        prev_keys = key_pressed

        # Ação do mouse
        dx = int(mouse_output[0] * mouse_norm["max_dx"])
        dy = int(mouse_output[1] * mouse_norm["max_dy"])
        if dx != 0 or dy != 0:
            move_mouse_relative(dx, dy)

        # Intervalo entre frames
        elapsed = time.time() - start
        if elapsed < INTERVAL:
            time.sleep(INTERVAL - elapsed)

except Exception as e:
    print(f"[ERRO] {e}")

finally:
    for key in KEY_NAMES:
        key_up(key)
    print("[INFO] Todos os comandos liberados.")
