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

# Configurações
IMAGE_DIR = "data/images"
LABEL_FILE = "data/labels.csv"
NORM_FILE = "data/mouse_normalization.json"
EPOCHS = 20
BATCH_SIZE = 16
SEQ_LEN = 6
FRAME_DELAY = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_memory_usage(note=""):
    mem = psutil.virtual_memory()
    print(f"[RAM] {note} - Uso: {mem.used // (1024 ** 2)} MB / {mem.total // (1024 ** 2)} MB")

# Transformações
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dataset com carregamento sob demanda
class VRChatImitationDataset(Dataset):
    def __init__(self, image_dir, label_file, transform, seq_len=3, frame_delay=5):
        self.image_dir = image_dir
        self.transform = transform
        self.seq_len = seq_len
        self.frame_delay = frame_delay

        with open(label_file, "r") as f:
            self.raw_data = list(csv.DictReader(f))

        self.max_index = len(self.raw_data) - frame_delay - (seq_len - 1)

        dx_vals, dy_vals = [], []
        for i in range(self.max_index):
            dx = float(self.raw_data[i + frame_delay]["mouse_dx"])
            dy = float(self.raw_data[i + frame_delay]["mouse_dy"])
            dx_vals.append(dx)
            dy_vals.append(dy)

        self.max_dx = max(1.0, max(abs(x) for x in dx_vals))
        self.max_dy = max(1.0, max(abs(y) for y in dy_vals))

        with open(NORM_FILE, "w") as f:
            json.dump({"max_dx": self.max_dx, "max_dy": self.max_dy}, f, indent=4)
        print(f"[INFO] Normalização salva em '{NORM_FILE}'")
        print_memory_usage("Final da preparação do dataset")

    def __len__(self):
        return self.max_index

    def __getitem__(self, idx):
        stacked_imgs = []
        for j in range(self.seq_len):
            image_path = os.path.join(self.image_dir, self.raw_data[idx + j]["image"])
            img = Image.open(image_path).convert("RGB")
            img_tensor = self.transform(img)
            stacked_imgs.append(img_tensor)

        stacked_tensor = torch.cat(stacked_imgs, dim=0)

        future = self.raw_data[idx + self.frame_delay]
        keys = future["keys"].split("+") if future["keys"] else []
        key_vec = [int(k in keys) for k in ["w", "s", "shift", "space", "a", "d"]]
        dx = float(future["mouse_dx"]) / self.max_dx
        dy = float(future["mouse_dy"]) / self.max_dy

        return stacked_tensor, torch.tensor(key_vec, dtype=torch.float32), torch.tensor([dx, dy], dtype=torch.float32)

def print_progress_bar(iteration, total, prefix='', suffix='', length=30):
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '#' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()

if __name__ == "__main__":
    print(f"Usando dispositivo: {DEVICE}")

    # Dataset e DataLoader otimizados
    dataset = VRChatImitationDataset(IMAGE_DIR, LABEL_FILE, transform, seq_len=SEQ_LEN, frame_delay=FRAME_DELAY)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

    # Modelos
    input_channels = 3 * SEQ_LEN
    keyboard_model = ImitationAgent(output_dim=6, mode="keyboard", input_channels=input_channels).to(DEVICE)
    mouse_model = ImitationAgent(output_dim=2, mode="mouse", input_channels=input_channels).to(DEVICE)

    k_optimizer = optim.Adam(keyboard_model.parameters(), lr=1e-4)
    m_optimizer = optim.Adam(mouse_model.parameters(), lr=1e-4)
    bce = nn.BCELoss()
    mse = nn.MSELoss()

    # Treinamento
    for epoch in range(EPOCHS):
        total_k_loss, total_m_loss = 0.0, 0.0
        num_batches = len(dataloader)

        for i, (imgs, key_labels, mouse_labels) in enumerate(dataloader, 1):
            imgs, key_labels, mouse_labels = imgs.to(DEVICE), key_labels.to(DEVICE), mouse_labels.to(DEVICE)

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

        print(f"\n[Epoch {epoch+1}] Keyboard Loss: {total_k_loss / num_batches:.4f} | Mouse Loss: {total_m_loss / num_batches:.4f}")
        print_memory_usage(f"Após epoch {epoch+1}")

    # Salvar modelos
    torch.save(keyboard_model.state_dict(), "imitation_keyboard_latest.pth")
    torch.save(mouse_model.state_dict(), "imitation_mouse_latest.pth")
    print("Modelos salvos com sucesso.")
