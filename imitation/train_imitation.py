# Treinamento por imitação a partir de imagens + CSV
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Parâmetros
IMAGE_DIR = "data/images"
LABEL_FILE = "data/labels.csv"
EPOCHS = 5
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset personalizado
class VRChatImitationDataset(Dataset):
    def __init__(self, image_dir, label_file, transform):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []
        with open(label_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                keys = row["keys"].split("+") if row["keys"] else []
                key_vector = [int(k in keys) for k in ["w", "s", "shift", "space", "a", "d"]]
                mouse_dx = float(row["mouse_dx"])
                mouse_dy = float(row["mouse_dy"])
                self.samples.append((row["image"], key_vector, [mouse_dx, mouse_dy]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, key_vec, mouse_vec = self.samples[idx]
        path = os.path.join(self.image_dir, filename)
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(key_vec, dtype=torch.float32), torch.tensor(mouse_vec, dtype=torch.float32)

# Modelos reutilizados do PPO
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

# Inicializar tudo
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = VRChatImitationDataset(IMAGE_DIR, LABEL_FILE, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

keyboard_model = KeyboardActor().to(DEVICE)
mouse_model = MouseActor().to(DEVICE)

k_optimizer = optim.Adam(keyboard_model.parameters(), lr=1e-4)
m_optimizer = optim.Adam(mouse_model.parameters(), lr=1e-4)

bce = nn.BCELoss()
mse = nn.MSELoss()

# Treinamento
for epoch in range(EPOCHS):
    total_k_loss = 0
    total_m_loss = 0
    for imgs, key_labels, mouse_labels in dataloader:
        imgs = imgs.to(DEVICE)
        key_labels = key_labels.to(DEVICE)
        mouse_labels = mouse_labels.to(DEVICE)

        k_optimizer.zero_grad()
        pred_keys = keyboard_model(imgs)
        k_loss = bce(pred_keys, key_labels)
        k_loss.backward()
        k_optimizer.step()

        m_optimizer.zero_grad()
        pred_mouse = mouse_model(imgs)
        m_loss = mse(pred_mouse, mouse_labels)
        m_loss.backward()
        m_optimizer.step()

        total_k_loss += k_loss.item()
        total_m_loss += m_loss.item()

    print(f"[Epoch {epoch+1}] Keyboard Loss: {total_k_loss:.4f} | Mouse Loss: {total_m_loss:.4f}")

# Salvar modelos
torch.save(keyboard_model.state_dict(), "imitation_keyboard_latest.pth")
torch.save(mouse_model.state_dict(), "imitation_mouse_latest.pth")
print("Modelos salvos com sucesso.")
