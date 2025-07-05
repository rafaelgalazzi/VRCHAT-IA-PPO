import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from torchvision import transforms
from agents.imitation_agent import ImitationAgentLSTM
from utils.input_controller import key_down, key_up, move_mouse_relative
import win32gui
import win32con


def get_vrchat_window_bbox():
    hwnd = win32gui.FindWindow(None, "VRChat")
    if hwnd == 0:
        print("[ERRO] Janela do VRChat não encontrada!")
        return None
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    win32gui.SetForegroundWindow(hwnd)
    return win32gui.GetWindowRect(hwnd)


# ======================
# Actor-Critic Models
# ======================

class KeyboardActorCritic(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.actor = ImitationAgentLSTM(output_dim=6, mode="keyboard", hidden_size=hidden_size)
        self.critic = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, seq):  # seq: [B, T, 3, H, W]
        probs = self.actor(seq)  # [B, 6]
        value = self.critic(probs.detach())  # [B, 1]
        return probs, value


class MouseActorCritic(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.actor = ImitationAgentLSTM(output_dim=2, mode="mouse", hidden_size=hidden_size)
        self.critic = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, seq):
        mean = self.actor(seq)
        std = torch.ones_like(mean) * 0.2
        value = self.critic(mean.detach())
        return mean, std, value


# ======================
# PPO Agent
# ======================

class VRChatAgent:
    def __init__(self, keyboard_model_path=None, mouse_model_path=None,
                 train_keyboard=True, train_mouse=True, enable_mouse_reset=False, stack_size=8):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_keyboard = train_keyboard
        self.train_mouse = train_mouse
        self.enable_mouse_reset = enable_mouse_reset
        self.stack_size = stack_size

        self.keyboard_model = KeyboardActorCritic().to(self.device)
        self.mouse_model = MouseActorCritic().to(self.device)

        self.k_opt = optim.Adam(self.keyboard_model.parameters(), lr=1e-4)
        self.m_opt = optim.Adam(self.mouse_model.parameters(), lr=1e-4)

        self.buffer = []
        self.last_mouse_move_time = time.time()
        self.last_mouse_pos = None

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.mouse_dx_max = 1.0
        self.mouse_dy_max = 1.0
        self._load_mouse_normalization()
        self._load_models(keyboard_model_path, mouse_model_path)

    def _load_mouse_normalization(self):
        path = "data/mouse_normalization.json"
        if os.path.exists(path):
            with open(path, "r") as f:
                norm = json.load(f)
                self.mouse_dx_max = norm.get("max_dx", 1.0)
                self.mouse_dy_max = norm.get("max_dy", 1.0)

    def _load_models(self, keyboard_path, mouse_path):
        if self.train_keyboard and keyboard_path and os.path.exists(keyboard_path):
            self.keyboard_model.actor.load_state_dict(torch.load(keyboard_path, map_location=self.device))
        if self.train_mouse and mouse_path and os.path.exists(mouse_path):
            self.mouse_model.actor.load_state_dict(torch.load(mouse_path, map_location=self.device))

    def act(self, stacked_seq):  # stacked_seq: [1, T, 3, 224, 224]
        key_action = key_logprob = key_value = None
        mouse_action = mouse_logprob = mouse_value = None

        if self.train_keyboard:
            probs, key_value = self.keyboard_model(stacked_seq)
            dist = torch.distributions.Bernoulli(probs)
            key_action = dist.sample()
            key_logprob = dist.log_prob(key_action)

        if self.train_mouse:
            mean, std, mouse_value = self.mouse_model(stacked_seq)
            dist = torch.distributions.Normal(mean, std)
            mouse_action = dist.sample()
            mouse_logprob = dist.log_prob(mouse_action)

        return (
            key_action.squeeze().cpu().numpy() if key_action is not None else None,
            mouse_action.squeeze().cpu().numpy() if mouse_action is not None else None,
            key_value, key_logprob, mouse_value, mouse_logprob
        )

    def apply_actions(self, key_action, mouse_action):
        keys = ["w", "s", "shift", "space", "a", "d"]
        for i, k in enumerate(keys):
            (key_down if key_action[i] > 0.5 else key_up)(k)
        if self.enable_mouse_reset and mouse_action is not None:
            dx = int(mouse_action[0] * self.mouse_dx_max)
            dy = int(mouse_action[1] * self.mouse_dy_max)
            move_mouse_relative(dx, dy)

    def store(self, seq, actions, k_r, m_r, k_lp, k_val, m_lp, m_val):
        self.buffer.append({
            "seq": seq,
            "key_action": actions[0],
            "mouse_action": actions[1],
            "key_reward": k_r,
            "mouse_reward": m_r,
            "key_logprob": k_lp,
            "mouse_logprob": m_lp,
            "key_value": k_val,
            "mouse_value": m_val
        })

    def update(self):
        if not self.buffer:
            return

        seqs = torch.stack([b["seq"].squeeze(0) for b in self.buffer]).to(self.device)  # [B, T, 3, 224, 224]

        if self.train_keyboard:
            actions = torch.tensor([b["key_action"] for b in self.buffer], dtype=torch.float32).to(self.device)
            rewards = torch.tensor([b["key_reward"] for b in self.buffer], dtype=torch.float32).to(self.device)
            logprobs_old = torch.stack([b["key_logprob"].squeeze() for b in self.buffer]).to(self.device)
            values_old = torch.cat([b["key_value"] for b in self.buffer]).squeeze()

            probs, values_new = self.keyboard_model(seqs)
            dist = torch.distributions.Bernoulli(probs)
            logprobs_new = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            advantage = rewards - values_old.detach()
            ratio = (logprobs_new - logprobs_old).exp()
            clipped = torch.clamp(ratio, 0.8, 1.2) * advantage
            policy_loss = -torch.min(ratio * advantage, clipped).mean()
            value_loss = 0.5 * (values_new.squeeze() - rewards).pow(2).mean()

            loss = policy_loss + value_loss - 0.01 * entropy
            self.k_opt.zero_grad()
            loss.backward()
            self.k_opt.step()

        if self.train_mouse:
            actions = torch.tensor([b["mouse_action"] for b in self.buffer], dtype=torch.float32).to(self.device)
            rewards = torch.tensor([b["mouse_reward"] for b in self.buffer], dtype=torch.float32).to(self.device)
            logprobs_old = torch.stack([b["mouse_logprob"].sum() for b in self.buffer]).to(self.device)
            values_old = torch.cat([b["mouse_value"] for b in self.buffer]).squeeze()

            mean, std, values_new = self.mouse_model(seqs)
            dist = torch.distributions.Normal(mean, std)
            logprobs_new = dist.log_prob(actions).sum(dim=1)
            entropy = dist.entropy().sum(dim=1).mean()

            advantage = rewards - values_old.detach()
            ratio = (logprobs_new - logprobs_old).exp()
            clipped = torch.clamp(ratio, 0.8, 1.2) * advantage
            policy_loss = -torch.min(ratio * advantage, clipped).mean()
            value_loss = 0.5 * (values_new.squeeze() - rewards).pow(2).mean()

            loss = policy_loss + value_loss - 0.01 * entropy
            self.m_opt.zero_grad()
            loss.backward()
            self.m_opt.step()

        self.buffer = []

    def get_keyboard_reward(self, img, key_action=None, yolo_result=None):
        reward = 0.0

        if yolo_result is not None:
            person_count, obstacle_detected, _ = yolo_result

            # Recompensa por tentar desviar se há obstáculo
            if obstacle_detected and key_action is not None:
                desvio = key_action[1] > 0.5 or key_action[4] > 0.5 or key_action[5] > 0.5  # s, a, d
                if desvio:
                    reward += 0.5
                if key_action[0] > 0.5:  # w
                    reward -= 0.2  # penalidade por ir de frente contra obstáculo

        return reward

    def get_mouse_reward(self, img, mouse_action=None, yolo_result=None):
        reward = 0.0
        if mouse_action is not None:
            moved = self.last_mouse_pos is None or np.linalg.norm(mouse_action - self.last_mouse_pos) > 0.01
            if moved:
                self.last_mouse_move_time = time.time()
                self.last_mouse_pos = mouse_action
            elif time.time() - self.last_mouse_move_time > 5:
                reward -= 0.5

        if yolo_result:
            _, _, bboxes = yolo_result
            if bboxes:
                center_x = img.width / 2
                centers = [(x1 + x2) / 2 for (x1, _, x2, _) in bboxes]
                avg_center = np.mean(centers)
                dist = abs(avg_center - center_x)
                reward += max(0, 1 - dist / (img.width / 2))
        return reward

    def save(self):
        if self.train_keyboard:
            torch.save(self.keyboard_model.state_dict(), "ppo_keyboard_model.pth")
        if self.train_mouse:
            torch.save(self.mouse_model.state_dict(), "ppo_mouse_model.pth")
