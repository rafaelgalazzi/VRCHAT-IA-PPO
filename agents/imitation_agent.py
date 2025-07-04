import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ImitationAgent(nn.Module):
    def __init__(self, output_dim, mode="keyboard", input_channels=3):
        super().__init__()
        self.mode = mode

        # Carrega modelo base
        base = models.resnet34(weights='IMAGENET1K_V1')

        # Modifica a primeira convolução se input_channels for diferente de 3
        if input_channels != 3:
            original_conv = base.conv1
            new_conv = nn.Conv2d(
                in_channels=input_channels,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )

            # Inicializa os primeiros 3 canais com os pesos existentes
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = original_conv.weight
                if input_channels > 3:
                    for i in range(3, input_channels):
                        new_conv.weight[:, i:i+1, :, :] = original_conv.weight[:, :1, :, :]
            
            base.conv1 = new_conv

        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 256)
        self.head = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        x = self.head(x)

        if self.mode == "keyboard":
            return torch.sigmoid(x)
        elif self.mode == "mouse":
            return x
        else:
            raise ValueError("Modo inválido: use 'keyboard' ou 'mouse'")
