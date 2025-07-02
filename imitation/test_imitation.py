# Test imitation model
import torch
import numpy as np
import time
import keyboard
from PIL import Image
from torchvision import transforms
from agents.imitation_agent import ImitationAgent
from utils.screen_utils import capture_vrchat_frame
from utils.input_controller import key_down, key_up, KEYS

MODEL_PATH = "imitation_keyboard_latest.pth"  # substitua se necessário
KEY_LIST = list(KEYS.keys())

model = ImitationAgent(output_dim=len(KEY_LIST)).cuda()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("[INFO] Pressione 'ESC' para sair.")

try:
    while True:
        if keyboard.is_pressed('esc'):
            break

        img = capture_vrchat_frame()
        if img is None:
            continue

        with torch.no_grad():
            tensor = transform(img).unsqueeze(0).cuda()
            output = torch.sigmoid(model(tensor)).cpu().numpy().squeeze()

        for i, key in enumerate(KEY_LIST):
            if output[i] > 0.5:
                key_down(key)
            else:
                key_up(key)

        time.sleep(0.05)

finally:
    for key in KEYS:
        key_up(key)
    print("[INFO] Teste finalizado.")
