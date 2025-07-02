import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models, transforms
from collections import deque
from utils.input_controller import key_down, key_up, move_mouse_relative

# Modelos
class KeyboardActorCritic(nn.Module):
    def __init__(self, num_keys=6):
        super().__init__()
        resnet = models.resnet34(weights='IMAGENET1K_V1')
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(resnet.fc.in_features, 256)
        self.actor = nn.Linear(256, num_keys)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = torch.relu(self.fc(x))
        return torch.sigmoid(self.actor(x)), self.critic(x)

class MouseActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet34(weights='IMAGENET1K_V1')
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(resnet.fc.in_features, 256)
        self.mean = nn.Linear(256, 2)
        self.std = nn.Linear(256, 2)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = torch.relu(self.fc(x))
        return torch.tanh(self.mean(x)), torch.nn.functional.softplus(self.std(x)) + 1e-4, self.critic(x)

# PPO Agente
class VRChatAgent:
    def __init__(self, keyboard_model_path=None, mouse_model_path=None,
                 train_keyboard=True, train_mouse=True, enable_mouse_reset=False):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_keyboard = train_keyboard
        self.train_mouse = train_mouse
        self.enable_mouse_reset = enable_mouse_reset

        self.keyboard_model = KeyboardActorCritic().to(self.device)
        self.mouse_model = MouseActorCritic().to(self.device)

        if keyboard_model_path and os.path.exists(keyboard_model_path):
            self.keyboard_model.load_state_dict(torch.load(keyboard_model_path, map_location=self.device))
        if mouse_model_path and os.path.exists(mouse_model_path):
            self.mouse_model.load_state_dict(torch.load(mouse_model_path, map_location=self.device))

        self.k_opt = optim.Adam(self.keyboard_model.parameters(), lr=1e-4)
        self.m_opt = optim.Adam(self.mouse_model.parameters(), lr=1e-4)

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.buffer = []

    def act(self, img):
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        key_action = key_log_prob = key_value = None
        mouse_action = mouse_log_prob = mouse_value = None

        if self.train_keyboard:
            key_probs, key_value = self.keyboard_model(tensor)
            key_dist = torch.distributions.Bernoulli(probs=key_probs)
            key_action = key_dist.sample()
            key_log_prob = key_dist.log_prob(key_action)

        if self.train_mouse:
            mean, std, mouse_value = self.mouse_model(tensor)
            dist = torch.distributions.Normal(mean, std)
            mouse_action = dist.sample()
            mouse_log_prob = dist.log_prob(mouse_action)

        return (
            key_action.squeeze().cpu().numpy() if key_action is not None else None,
            mouse_action.squeeze().cpu().numpy() if mouse_action is not None else None,
            key_value, key_log_prob, mouse_value, mouse_log_prob
        )

    def store(self, img, actions, k_r, m_r, k_lp, k_val, m_lp, m_val):
        self.buffer.append({
            "img": img,
            "key_action": actions[0], "mouse_action": actions[1],
            "key_logprob": k_lp, "mouse_logprob": m_lp,
            "key_value": k_val, "mouse_value": m_val,
            "key_reward": k_r, "mouse_reward": m_r
        })

    def update(self):
        if not self.buffer:
            return

        imgs = torch.stack([self.transform(b["img"]) for b in self.buffer]).to(self.device)

        if self.train_keyboard:
            actions = torch.tensor([b["key_action"] for b in self.buffer]).float().to(self.device)
            rewards = torch.tensor([b["key_reward"] for b in self.buffer]).float().to(self.device)
            logprobs = torch.stack([b["key_logprob"].squeeze() for b in self.buffer]).to(self.device)
            values = torch.cat([b["key_value"] for b in self.buffer]).squeeze().to(self.device)

            probs, values_pred = self.keyboard_model(imgs)
            dist = torch.distributions.Bernoulli(probs)
            entropy = dist.entropy().mean()

            new_logprobs = dist.log_prob(actions)
            advantage = rewards - values.squeeze().detach()
            ratio = (new_logprobs - logprobs).exp()
            clip = torch.clamp(ratio, 0.8, 1.2) * advantage

            k_loss = -torch.min(ratio * advantage, clip).mean() + 0.5 * (values_pred.squeeze() - rewards).pow(2).mean() - 0.01 * entropy
            self.k_opt.zero_grad()
            k_loss.backward()
            self.k_opt.step()

        if self.train_mouse:
            actions = torch.tensor([b["mouse_action"] for b in self.buffer]).float().to(self.device)
            rewards = torch.tensor([b["mouse_reward"] for b in self.buffer]).float().to(self.device)
            logprobs = torch.stack([b["mouse_logprob"].sum() for b in self.buffer]).to(self.device)
            values = torch.cat([b["mouse_value"] for b in self.buffer]).squeeze().to(self.device)

            mean, std, values_pred = self.mouse_model(imgs)
            dist = torch.distributions.Normal(mean, std)
            entropy = dist.entropy().sum(dim=1).mean()

            new_logprobs = dist.log_prob(actions).sum(dim=1)
            advantage = rewards - values.detach()
            ratio = (new_logprobs - logprobs).exp()
            clip = torch.clamp(ratio, 0.8, 1.2) * advantage

            m_loss = -torch.min(ratio * advantage, clip).mean() + 0.5 * (values_pred.squeeze() - rewards).pow(2).mean() - 0.01 * entropy
            self.m_opt.zero_grad()
            m_loss.backward()
            self.m_opt.step()

        self.buffer = []

    def apply_actions(self, key_action, mouse_action):
        keys = ["w", "s", "shift", "space", "a", "d"]
        for i, k in enumerate(keys):
            (key_down if key_action[i] > 0.5 else key_up)(k)
        dx, dy = mouse_action * 30
        move_mouse_relative(dx, dy)

    def get_keyboard_reward(self, img, key_action, yolo_result):
        person_count, obstacle_detected, _ = yolo_result
        reward = 0.0
        if person_count > 0:
            reward += 1.0
        if obstacle_detected:
            reward -= 0.5
        return reward

    def get_mouse_reward(self, img, mouse_action, yolo_result):
        _, _, bboxes = yolo_result
        reward = 0.0
        if bboxes:
            center_x = img.width / 2
            target_xs = [(x1 + x2) / 2 for (x1, y1, x2, y2) in bboxes]
            avg_x = np.mean(target_xs)
            reward += max(0, 1 - abs(avg_x - center_x) / (img.width / 2))  # mais perto do centro, maior a recompensa
        return reward

    def save(self):
        if self.train_keyboard:
            torch.save(self.keyboard_model.state_dict(), "ppo_keyboard_model.pth")
        if self.train_mouse:
            torch.save(self.mouse_model.state_dict(), "ppo_mouse_model.pth")
