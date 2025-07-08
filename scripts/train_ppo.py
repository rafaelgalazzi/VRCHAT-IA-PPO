import time
import os
import json
import sys
from collections import deque
from PIL import Image
import numpy as np
import torch
import keyboard
from mss import mss
import psutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.ppo_agent import VRChatAgent, get_vrchat_window_bbox
from utils.yolo_utils import analyze_image_with_yolo, load_yolo_model

# --- CONFIG ---
EPISODES = 10000
STEPS_PER_EPISODE = 500
STEP_DELAY = 1.0 / 20  # 20 FPS
TRAIN_KEYBOARD = True
TRAIN_MOUSE = True
YOLO_INTERVAL = 10
FRAME_STACK = 6
UPDATE_INTERVAL = 100

# Model paths
KEYBOARD_MODEL_PATH = "ppo_keyboard_model.pth"
MOUSE_MODEL_PATH = "ppo_mouse_model.pth"
IMITATION_KEYBOARD_PATH = "imitation_keyboard_latest.pth"
IMITATION_MOUSE_PATH = "imitation_mouse_latest.pth"

# YOLO
yolo_model = load_yolo_model("yolov8n.pt")

# Mouse normalization
NORMALIZATION_PATH = "data/mouse_normalization.json"
mouse_norm = {"max_dx": 1.0, "max_dy": 1.0}
if os.path.exists(NORMALIZATION_PATH):
    with open(NORMALIZATION_PATH, "r") as f:
        mouse_norm = json.load(f)
    print(f"[INFO] Mouse normalization loaded: {mouse_norm}")
else:
    print(f"[WARNING] Normalization file not found: {NORMALIZATION_PATH}")

# Transform
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Screen capture
rect = get_vrchat_window_bbox()
if not rect:
    print("[ERROR] VRChat window not found.")
    sys.exit()
monitor = {"top": rect[1], "left": rect[0], "width": rect[2] - rect[0], "height": rect[3] - rect[1]}


def capture_frame():
    with mss() as sct:
        img = sct.grab(monitor)
        return Image.frombytes("RGB", img.size, img.rgb)


def train():
    keyboard_model_path = KEYBOARD_MODEL_PATH if os.path.exists(KEYBOARD_MODEL_PATH) else IMITATION_KEYBOARD_PATH
    mouse_model_path = MOUSE_MODEL_PATH if os.path.exists(MOUSE_MODEL_PATH) else IMITATION_MOUSE_PATH

    agent = VRChatAgent(
        keyboard_model_path=keyboard_model_path,
        mouse_model_path=mouse_model_path,
        train_keyboard=TRAIN_KEYBOARD,
        train_mouse=TRAIN_MOUSE,
    )

    frame_buffer = deque(maxlen=FRAME_STACK)

    print("[INFO] Starting training... Press ESC to stop.")

    for episode in range(EPISODES):
        if keyboard.is_pressed("esc"):
            print("[INFO] Interrupted before episode. Saving...")
            agent.save()
            break

        total_reward = 0
        steps_taken = 0

        while steps_taken < STEPS_PER_EPISODE:
            if keyboard.is_pressed("esc"):
                print("[INFO] Interrupted mid-episode. Saving...")
                agent.save()
                return

            img = capture_frame()
            if img is None:
                continue

            frame_buffer.append(transform(img))

            if len(frame_buffer) < FRAME_STACK:
                time.sleep(STEP_DELAY)
                continue

            sequence = torch.stack(list(frame_buffer), dim=0).unsqueeze(0).to(agent.device)

            yolo_result = analyze_image_with_yolo(yolo_model, img) if steps_taken % YOLO_INTERVAL == 0 else None

            key_action, mouse_action, key_val, key_logp, mouse_val, mouse_logp = agent.act(sequence)

            if key_action is None and mouse_action is None:
                time.sleep(STEP_DELAY)
                continue

            if mouse_action is not None:
                mouse_action = np.array([
                    np.clip(mouse_action[0] / mouse_norm["max_dx"], -1, 1),
                    np.clip(mouse_action[1] / mouse_norm["max_dy"], -1, 1)
                ])

            k_reward = agent.get_keyboard_reward(img, key_action, yolo_result) if TRAIN_KEYBOARD else None
            m_reward = agent.get_mouse_reward(img, mouse_action, yolo_result) if TRAIN_MOUSE else None
            total_reward += (k_reward or 0) + (m_reward or 0)

            agent.store(
                sequence.squeeze(0).detach(),
                (key_action, mouse_action),
                k_reward,
                m_reward,
                key_logp,
                key_val,
                mouse_logp,
                mouse_val
            )

            denorm_mouse = None
            if mouse_action is not None:
                denorm_mouse = np.array([
                    mouse_action[0] * mouse_norm["max_dx"],
                    mouse_action[1] * mouse_norm["max_dy"]
                ])
            agent.apply_actions(key_action, denorm_mouse)

            if steps_taken > 0 and steps_taken % UPDATE_INTERVAL == 0:
                agent.update()
                torch.cuda.empty_cache()

            steps_taken += 1

            cuda_mem = torch.cuda.memory_allocated(agent.device) / (1024 ** 3)
            print(f"[DEBUG] EP {episode + 1} STEP {steps_taken} | Buffer: {len(agent.buffer)} | CUDA Mem: {cuda_mem:.2f} GB")
            time.sleep(STEP_DELAY)

        print(f"[EP {episode + 1}] Total reward: {total_reward:.2f} | Steps: {steps_taken}")
        agent.update()
        agent.save()


if __name__ == "__main__":
    train()
