import time
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import ImageGrab
import win32gui
import win32con
import numpy as np
import json
import os

from utils.input_controller import key_down, key_up, move_mouse_relative, KEYS

# Configurações
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
FPS = 10
INTERVAL = 1.0 / FPS
KEY_NAMES = ["w", "s", "shift", "space", "a", "d"]

# Normalização
NORMALIZATION_PATH = "data/mouse_normalization.json"
mouse_norm = {"max_dx": 1.0, "max_dy": 1.0}
if os.path.exists(NORMALIZATION_PATH):
    with open(NORMALIZATION_PATH, "r") as f:
        mouse_norm = json.load(f)
    print(f"[INFO] Normalização de mouse carregada: {mouse_norm}")
else:
    print(f"[AVISO] Arquivo de normalização não encontrado: {NORMALIZATION_PATH}")

# Modelos
class KeyboardActor(nn.Module):
    def __init__(self):
        super().__init__()
        base = nn.Sequential(*list(torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True).children())[:-1])
        self.backbone = base
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 256)
        self.head = nn.Linear(256, 6)

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = torch.relu(self.fc(x))
        return torch.sigmoid(self.head(x))

class MouseActor(nn.Module):
    def __init__(self):
        super().__init__()
        base = nn.Sequential(*list(torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True).children())[:-1])
        self.backbone = base
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 256)
        self.head = nn.Linear(256, 2)

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = torch.relu(self.fc(x))
        return self.head(x)

# Transformação da imagem
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Captura da janela do VRChat
def get_vrchat_window_bbox():
    hwnd = win32gui.FindWindow(None, "VRChat")
    if hwnd == 0:
        print("[ERRO] Janela do VRChat não encontrada!")
        return None
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    win32gui.SetForegroundWindow(hwnd)
    return win32gui.GetWindowRect(hwnd)

# Inicialização dos modelos
keyboard_model = KeyboardActor().to(DEVICE)
mouse_model = MouseActor().to(DEVICE)
keyboard_model.load_state_dict(torch.load("imitation_keyboard_latest.pth", map_location=DEVICE))
mouse_model.load_state_dict(torch.load("imitation_mouse_latest.pth", map_location=DEVICE))
keyboard_model.eval()
mouse_model.eval()

bbox = get_vrchat_window_bbox()
if not bbox:
    exit(1)

print("[INFO] IA rodando... Pressione Ctrl+C para interromper.")
prev_keys = np.zeros(len(KEY_NAMES))

try:
    while True:
        start = time.time()

        img = ImageGrab.grab(bbox=bbox).resize((IMAGE_SIZE, IMAGE_SIZE))
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            key_output = keyboard_model(img_tensor).cpu().numpy()[0]
            mouse_output = mouse_model(img_tensor).cpu().numpy()[0]

        # Desnormaliza o movimento do mouse
        dx = int(mouse_output[0] * mouse_norm["max_dx"])
        dy = int(mouse_output[1] * mouse_norm["max_dy"])

        key_pressed = (key_output > 0.5).astype(int)

        for i, key_name in enumerate(KEY_NAMES):
            if key_pressed[i] and not prev_keys[i]:
                key_down(key_name)
            elif not key_pressed[i] and prev_keys[i]:
                key_up(key_name)
        prev_keys = key_pressed

        if dx != 0 or dy != 0:
            move_mouse_relative(dx, dy)

        elapsed = time.time() - start
        if elapsed < INTERVAL:
            time.sleep(INTERVAL - elapsed)

except KeyboardInterrupt:
    print("\n[INFO] Execução encerrada. Liberando teclas...")
    for key in KEY_NAMES:
        key_up(key)
