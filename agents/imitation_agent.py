import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ImitationAgentLSTM(nn.Module):
    def __init__(self, output_dim, mode="keyboard", hidden_size=256):
        super().__init__()
        self.mode = mode

        base = models.resnet34(weights='IMAGENET1K_V1')
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])  # Remove fc
        self.feature_dim = 512  # ResNet34 output

        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, 256)
        self.head = nn.Linear(256, output_dim)

    def forward(self, x):  # x: [B, T, 3, H, W]
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        feats = self.feature_extractor(x).view(B, T, -1)  # [B, T, 512]

        lstm_out, _ = self.lstm(feats)  # [B, T, H]
        last = lstm_out[:, -1]  # Última saída

        x = F.relu(self.fc(last))
        x = self.head(x)

        if self.mode == "keyboard":
            return torch.sigmoid(x)
        elif self.mode == "mouse":
            return x
        else:
            raise ValueError("Modo inválido")
