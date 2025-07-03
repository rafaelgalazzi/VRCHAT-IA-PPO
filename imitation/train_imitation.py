import os
import csv
import json
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys

# Imports internos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agents.imitation_agent import ImitationAgent

# Config
IMAGE_DIR = "data/images"
LABEL_FILE = "data/labels.csv"
NORM_FILE = "data/mouse_normalization.json"
EPOCHS = 20
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {DEVICE}")

def print_memory_usage(note=""):
    mem = psutil.virtual_memory()
    print(f"[RAM] {note} - Uso: {mem.used // (1024 ** 2)} MB / {mem.total // (1024 ** 2)} MB")

# Dataset com cache
class VRChatImitationDataset(Dataset):
    def __init__(self, image_dir, label_file, transform, frame_delay=5):
        self.samples = []
        all_dx, all_dy = [], []

        with open(label_file, "r") as f:
            reader = csv.DictReader(f)
            raw_data = list(reader)

        for i in range(len(raw_data) - frame_delay):
            image_file = raw_data[i]["image"]
            future = raw_data[i + frame_delay]

            keys = future["keys"].split("+") if future["keys"] else []
            key_vec = [int(k in keys) for k in ["w", "s", "shift", "space", "a", "d"]]
            dx = float(future["mouse_dx"])
            dy = float(future["mouse_dy"])

            path = os.path.join(image_dir, image_file)
            img = Image.open(path).convert("RGB")
            img = transform(img)

            self.samples.append((img, torch.tensor(key_vec, dtype=torch.float32), torch.tensor([dx, dy], dtype=torch.float32)))
            all_dx.append(dx)
            all_dy.append(dy)

            if i % 5000 == 0:
                print_memory_usage(f"Ao carregar {i} imagens")

        self.max_dx = max(1.0, max(abs(x) for x in all_dx))
        self.max_dy = max(1.0, max(abs(y) for y in all_dy))

        with open(NORM_FILE, "w") as f:
            json.dump({"max_dx": self.max_dx, "max_dy": self.max_dy}, f, indent=4)
        print(f"[INFO] Normalização salva em '{NORM_FILE}'")
        print_memory_usage("Final do carregamento do dataset")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, key_vec, mouse_vec = self.samples[idx]
        dx, dy = mouse_vec
        dx /= self.max_dx
        dy /= self.max_dy
        return img, key_vec, torch.tensor([dx, dy], dtype=torch.float32)

# Transformações
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dataset e DataLoader
dataset = VRChatImitationDataset(IMAGE_DIR, LABEL_FILE, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Modelos unificados
keyboard_model = ImitationAgent(output_dim=6, mode="keyboard").to(DEVICE)
mouse_model = ImitationAgent(output_dim=2, mode="mouse").to(DEVICE)

k_optimizer = optim.Adam(keyboard_model.parameters(), lr=1e-4)
m_optimizer = optim.Adam(mouse_model.parameters(), lr=1e-4)

bce = nn.BCELoss()
mse = nn.MSELoss()

def print_progress_bar(iteration, total, prefix='', suffix='', length=30):
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '#' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()

# Treinamento
for epoch in range(EPOCHS):
    total_k_loss = 0
    total_m_loss = 0
    num_batches = len(dataloader)

    for i, (imgs, key_labels, mouse_labels) in enumerate(dataloader, 1):
        imgs = imgs.to(DEVICE)
        key_labels = key_labels.to(DEVICE)
        mouse_labels = mouse_labels.to(DEVICE)

        # Teclado
        k_optimizer.zero_grad()
        pred_keys = keyboard_model(imgs)
        k_loss = bce(pred_keys, key_labels)
        k_loss.backward()
        k_optimizer.step()
        total_k_loss += k_loss.item()

        # Mouse
        m_optimizer.zero_grad()
        pred_mouse = mouse_model(imgs)
        m_loss = mse(pred_mouse, mouse_labels)
        m_loss.backward()
        m_optimizer.step()
        total_m_loss += m_loss.item()

        print_progress_bar(i, num_batches, prefix=f"Epoch {epoch+1}", suffix="Processando", length=40)

    avg_k_loss = total_k_loss / num_batches
    avg_m_loss = total_m_loss / num_batches
    print(f"\n[Epoch {epoch+1}] Keyboard Loss: {avg_k_loss:.4f} | Mouse Loss: {avg_m_loss:.4f}")
    print_memory_usage(f"Após epoch {epoch+1}")

# Salvar modelos
torch.save(keyboard_model.state_dict(), "imitation_keyboard_latest.pth")
torch.save(mouse_model.state_dict(), "imitation_mouse_latest.pth")
print("Modelos salvos com sucesso.")
