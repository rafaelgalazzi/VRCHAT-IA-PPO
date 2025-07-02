import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ImitationAgent(nn.Module):
    def __init__(self, output_dim, mode="keyboard"):
        """
        mode: 'keyboard' for binary outputs with sigmoid,
              'mouse' for continuous outputs (dx, dy)
        """
        super().__init__()
        self.mode = mode
        base = models.resnet34(weights='IMAGENET1K_V1')
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(base.fc.in_features, 128)
        self.out = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        x = self.out(x)

        if self.mode == "keyboard":
            return torch.sigmoid(x)
        elif self.mode == "mouse":
            return x  # ou torch.tanh(x) para limitar o range
        else:
            raise ValueError("Modo inválido: use 'keyboard' ou 'mouse'")
