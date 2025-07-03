import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ImitationAgent(nn.Module):
    def __init__(self, output_dim, mode="keyboard"):
        super().__init__()
        self.mode = mode
        base = models.resnet34(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 256)  # Compatível com seu treino
        self.head = nn.Linear(256, output_dim)  # Corrigido aqui

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        x = self.head(x)  # Corrigido aqui também

        if self.mode == "keyboard":
            return torch.sigmoid(x)
        elif self.mode == "mouse":
            return x
        else:
            raise ValueError("Modo inválido: use 'keyboard' ou 'mouse'")

