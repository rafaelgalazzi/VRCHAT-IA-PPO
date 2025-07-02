# PPO agent completo com carregamento de pesos de imitação e funções de recompensa
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models, transforms
from collections import deque
from utils.input_controller import key_down, key_up, move_mouse_relative
import csv

class KeyboardActorCritic(nn.Module):
    def __init__(self, num_keys):
        super().__init__()
        base = models.resnet34(weights='IMAGENET1K_V1')
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(base.fc.in_features, 256)
        self.actor_keys = nn.Linear(256, num_keys)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = torch.relu(self.fc(x))
        key_out = torch.sigmoid(self.actor_keys(x))
        value = self.critic(x)
        return key_out, value

class MouseActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet34(weights='IMAGENET1K_V1')
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(base.fc.in_features, 256)
        self.actor_mouse = nn.Linear(256, 2)
        self.actor_std = nn.Linear(256, 2)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = torch.relu(self.fc(x))
        mouse_mean = torch.tanh(self.actor_mouse(x))
        mouse_std = torch.nn.functional.softplus(self.actor_std(x)) + 1e-4
        value = self.critic(x)
        return mouse_mean, mouse_std, value

class VRChatAgent:
    def __init__(self, num_keys=6, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.keyboard_model = KeyboardActorCritic(num_keys).to(self.device)
        self.mouse_model = MouseActorCritic().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self._load_if_imitation_exists()

    def _load_if_imitation_exists(self):
        if os.path.exists("imitation_keyboard_latest.pth"):
            print("[INFO] Carregando pesos do teclado de imitação...")
            self.keyboard_model.load_state_dict(torch.load("imitation_keyboard_latest.pth", map_location=self.device), strict=False)
        if os.path.exists("imitation_mouse_latest.pth"):
            print("[INFO] Carregando pesos do mouse de imitação...")
            self.mouse_model.load_state_dict(torch.load("imitation_mouse_latest.pth", map_location=self.device), strict=False)

    def act(self, img):
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            key_probs, _ = self.keyboard_model(img_tensor)
            mouse_mean, mouse_std, _ = self.mouse_model(img_tensor)

        key_dist = torch.distributions.Bernoulli(key_probs)
        mouse_dist = torch.distributions.Normal(mouse_mean, mouse_std)

        key_action = key_dist.sample().squeeze().cpu().numpy()
        mouse_action = mouse_dist.sample().squeeze().cpu().numpy()

        return key_action, mouse_action

    def apply_actions(self, key_action, mouse_action):
        for i, k in enumerate(["w", "s", "shift", "space", "a", "d"]):
            if key_action[i] > 0.5:
                key_down(k)
            else:
                key_up(k)

        dx, dy = mouse_action * 30  # scaling factor
        move_mouse_relative(dx, dy)

    def get_keyboard_reward(self, img, key_action, yolo_result):
        person_count, obstacle_detected, _ = yolo_result
        reward = 0
        if person_count > 0:
            reward += 1.0
        if obstacle_detected:
            reward -= 0.5
        return reward

    def get_mouse_reward(self, img, mouse_action, yolo_result):
        _, _, bboxes = yolo_result
        reward = 0
        if bboxes:
            center_x = img.width / 2
            person_center = np.mean([[(box[0]+box[2])/2 for box in bboxes]])
            if abs(person_center - center_x) < 50:
                reward += 0.5
        return reward
