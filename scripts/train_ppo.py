import time
import os
import json
import sys
from threading import Thread
from queue import Queue
from collections import deque
from PIL import Image
import numpy as np
import torch
import keyboard  # <- Novo para parar com ESC
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.ppo_agent import VRChatAgent, get_vrchat_window_bbox
from utils.yolo_utils import analyze_image_with_yolo, load_yolo_model
from mss import mss

# --- CONFIG ---
EPISODES = 10000
STEPS_PER_EPISODE = 500
STEP_DELAY = 1.0 / 20  # 20 FPS
MOUSE_RESET = True
TRAIN_KEYBOARD = True
TRAIN_MOUSE = False
YOLO_INTERVAL = 10
FRAME_STACK = 6

# Caminhos dos modelos
KEYBOARD_MODEL_PATH = "ppo_keyboard_model.pth"
MOUSE_MODEL_PATH = "ppo_mouse_model.pth"
IMITATION_KEYBOARD_PATH = "imitation_keyboard_latest.pth"
IMITATION_MOUSE_PATH = "imitation_mouse_latest.pth"

# YOLO
yolo_model = load_yolo_model("yolov8n.pt")

# Normalização do mouse
NORMALIZATION_PATH = "data/mouse_normalization.json"
mouse_norm = {"max_dx": 1.0, "max_dy": 1.0}
if os.path.exists(NORMALIZATION_PATH):
    with open(NORMALIZATION_PATH, "r") as f:
        mouse_norm = json.load(f)
    print(f"[INFO] Normalização de mouse carregada: {mouse_norm}")
else:
    print(f"[AVISO] Arquivo de normalização não encontrado: {NORMALIZATION_PATH}")

# Frame queue
frame_queue = Queue(maxsize=10)

def capture_thread():
    sct = mss()
    rect = get_vrchat_window_bbox()
    if not rect:
        print("[ERRO] Janela do VRChat não encontrada.")
        return
    monitor = {"top": rect[1], "left": rect[0], "width": rect[2] - rect[0], "height": rect[3] - rect[1]}
    while True:
        img = sct.grab(monitor)
        img_pil = Image.frombytes("RGB", img.size, img.rgb).resize((224, 224), Image.Resampling.LANCZOS)
        frame_queue.put(img_pil)
        time.sleep(STEP_DELAY)

def train():
    # Escolhe modelos de entrada
    keyboard_model_path = None
    mouse_model_path = None
    if TRAIN_KEYBOARD:
        if os.path.exists(KEYBOARD_MODEL_PATH):
            keyboard_model_path = KEYBOARD_MODEL_PATH
        elif os.path.exists(IMITATION_KEYBOARD_PATH):
            keyboard_model_path = IMITATION_KEYBOARD_PATH
    if TRAIN_MOUSE:
        if os.path.exists(MOUSE_MODEL_PATH):
            mouse_model_path = MOUSE_MODEL_PATH
        elif os.path.exists(IMITATION_MOUSE_PATH):
            mouse_model_path = IMITATION_MOUSE_PATH

    agent = VRChatAgent(
        keyboard_model_path=keyboard_model_path,
        mouse_model_path=mouse_model_path,
        train_keyboard=TRAIN_KEYBOARD,
        train_mouse=TRAIN_MOUSE,
        enable_mouse_reset=MOUSE_RESET
    )

    frame_stack = deque(maxlen=FRAME_STACK)
    Thread(target=capture_thread, daemon=True).start()
    print("[INFO] Iniciando captura e treinamento... Pressione ESC para parar.")

    for episode in range(EPISODES):
        if keyboard.is_pressed("esc"):
            print("[INFO] Treinamento interrompido antes do episódio. Salvando...")
            agent.save()
            break

        total_reward = 0
        steps_taken = 0

        for step in range(STEPS_PER_EPISODE):
            if keyboard.is_pressed("esc"):
                print("[INFO] Treinamento interrompido no meio do episódio. Salvando...")
                agent.save()
                return

            if frame_queue.empty():
                time.sleep(0.01)
                continue

            raw_frame = frame_queue.get()
            frame_stack.append(raw_frame)

            if len(frame_stack) < FRAME_STACK:
                continue

            # Empilha e transforma
            stacked_tensor = torch.stack([agent.transform(f) for f in frame_stack])  # [T, 3, 224, 224]
            stacked_tensor = stacked_tensor.unsqueeze(0).to(agent.device)  # [1, T, 3, 224, 224]

            yolo_result = analyze_image_with_yolo(yolo_model, frame_stack[-1]) if step % YOLO_INTERVAL == 0 else None

            key_action, mouse_action, key_val, key_logp, mouse_val, mouse_logp = agent.act(stacked_tensor)
            if key_action is None and mouse_action is None:
                continue

            if mouse_action is not None:
                mouse_action = np.array([
                    np.clip(mouse_action[0] / mouse_norm["max_dx"], -1, 1),
                    np.clip(mouse_action[1] / mouse_norm["max_dy"], -1, 1)
                ])

            # Recompensa
            k_reward = agent.get_keyboard_reward(frame_stack[-1], key_action, yolo_result) if TRAIN_KEYBOARD else None
            m_reward = agent.get_mouse_reward(frame_stack[-1], mouse_action, yolo_result) if TRAIN_MOUSE else None
            total_reward += (k_reward or 0) + (m_reward or 0)

            agent.store(
                stacked_tensor.squeeze(0),
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

            if steps_taken > 0 and steps_taken % 100 == 0:
                agent.update()
                agent.save()

            steps_taken += 1
            time.sleep(STEP_DELAY)

        print(f"[EP {episode + 1}] Total reward: {total_reward:.2f} | Steps: {steps_taken}")
        agent.update()
        agent.save()

if __name__ == "__main__":
    train()
