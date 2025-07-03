import os
import csv
import json
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34, ResNet34_Weights
import numpy as np
import sys

# Parâmetros
IMAGE_DIR = "data/images"
LABEL_FILE = "data/labels.csv"
NORM_FILE = "data/mouse_normalization.json"
EPOCHS = 20
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {DEVICE}")

# Dataset personalizado
class VRChatImitationDataset(Dataset):
    def __init__(self, image_dir, label_file, transform, frame_delay=5):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []
        all_dx, all_dy = [], []

        # Carrega todos os dados
        raw_data = []
        with open(label_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_data.append(row)

        # Aplica o deslocamento (desconsidera os últimos N frames)
        for i in range(len(raw_data) - frame_delay):
            image = raw_data[i]["image"]
            future = raw_data[i + frame_delay]  # Ação futura

            keys = future["keys"].split("+") if future["keys"] else []
            key_vector = [int(k in keys) for k in ["w", "s", "shift", "space", "a", "d"]]
            mouse_dx = float(future["mouse_dx"])
            mouse_dy = float(future["mouse_dy"])

            self.samples.append((image, key_vector, [mouse_dx, mouse_dy]))
            all_dx.append(mouse_dx)
            all_dy.append(mouse_dy)

        # Normalização dos valores de mouse
        self.max_dx = max(1.0, max(abs(x) for x in all_dx))
        self.max_dy = max(1.0, max(abs(y) for y in all_dy))

        with open(NORM_FILE, "w") as f:
            json.dump({"max_dx": self.max_dx, "max_dy": self.max_dy}, f, indent=4)
        print(f"[INFO] Normalização salva em '{NORM_FILE}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, key_vec, mouse_vec = self.samples[idx]
        path = os.path.join(self.image_dir, filename)
        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        dx, dy = mouse_vec
        dx /= self.max_dx
        dy /= self.max_dy

        return img, torch.tensor(key_vec, dtype=torch.float32), torch.tensor([dx, dy], dtype=torch.float32)

# Modelos
class KeyboardActor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1
        base_model = resnet34(weights=weights)
        base = nn.Sequential(*list(base_model.children())[:-1])
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
        weights = ResNet34_Weights.IMAGENET1K_V1
        base_model = resnet34(weights=weights)
        base = nn.Sequential(*list(base_model.children())[:-1])
        self.backbone = base
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 256)
        self.head = nn.Linear(256, 2)

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = torch.relu(self.fc(x))
        return self.head(x)

# Transformação
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset e DataLoader
dataset = VRChatImitationDataset(IMAGE_DIR, LABEL_FILE, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

keyboard_model = KeyboardActor().to(DEVICE)
mouse_model = MouseActor().to(DEVICE)

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
        print()  # quebra de linha no fim da barra

# Treinamento
for epoch in range(EPOCHS):
    total_k_loss = 0
    total_m_loss = 0
    num_batches = len(dataloader)

    for i, (imgs, key_labels, mouse_labels) in enumerate(dataloader, start=1):
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

        # Atualiza barra de progresso
        print_progress_bar(i, num_batches, prefix=f"Epoch {epoch+1}", suffix="Processando", length=40)

    avg_k_loss = total_k_loss / num_batches
    avg_m_loss = total_m_loss / num_batches
    print(f"[Epoch {epoch+1}] Keyboard Loss: {avg_k_loss:.4f} | Mouse Loss: {avg_m_loss:.4f}")

# Salvar modelos
torch.save(keyboard_model.state_dict(), "imitation_keyboard_latest.pth")
torch.save(mouse_model.state_dict(), "imitation_mouse_latest.pth")
print("Modelos salvos com sucesso.")
