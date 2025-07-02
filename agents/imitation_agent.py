# Imitation learning agent
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ImitationAgent(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        base = models.resnet34(weights='IMAGENET1K_V1')
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(base.fc.in_features, 128)
        self.out = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        return self.out(x)

# Para teclado: output_dim = número de teclas (com sigmoide binária)
# Para mouse: output_dim = 2 (dx, dy contínuos)