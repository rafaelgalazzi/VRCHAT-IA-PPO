import time
from threading import Thread
from queue import Queue
from PIL import Image
import numpy as np

from agents.ppo_agent import VRChatAgent, get_vrchat_window_bbox, compute_proximity_reward
from utils.yolo_utils import analyze_image_with_yolo
from mss import mss

# --- CONFIG ---
EPISODES = 10000
STEPS_PER_EPISODE = 500
STEP_DELAY = 1.0 / 20  # 20 FPS
MOUSE_RESET = True
TRAIN_KEYBOARD = True
TRAIN_MOUSE = False

# Model paths
KEYBOARD_MODEL_PATH = "ppo_keyboard_model.pth"
MOUSE_MODEL_PATH = "ppo_mouse_model.pth"

# --- FRAME QUEUE + CAPTURE THREAD ---
frame_queue = Queue(maxsize=10)

def capture_thread():
    sct = mss()
    rect = get_vrchat_window_bbox()
    if not rect:
        print("VRChat window not found.")
        return
    monitor = {"top": rect[1], "left": rect[0], "width": rect[2] - rect[0], "height": rect[3] - rect[1]}
    while True:
        img = sct.grab(monitor)
        img_pil = Image.frombytes("RGB", img.size, img.rgb).resize((224, 224), Image.Resampling.LANCZOS)
        frame_queue.put(img_pil)
        time.sleep(STEP_DELAY)

# --- MAIN TRAIN FUNCTION ---
def train():
    agent = VRChatAgent(
        keyboard_model_path=KEYBOARD_MODEL_PATH if TRAIN_KEYBOARD else None,
        mouse_model_path=MOUSE_MODEL_PATH if TRAIN_MOUSE else None,
        train_keyboard=TRAIN_KEYBOARD,
        train_mouse=TRAIN_MOUSE,
        enable_mouse_reset=MOUSE_RESET
    )

    print("Iniciando captura e treinamento...")
    Thread(target=capture_thread, daemon=True).start()

    for episode in range(EPISODES):
        total_reward = 0
        steps_taken = 0

        for step in range(STEPS_PER_EPISODE):
            if frame_queue.empty():
                time.sleep(0.01)
                continue

            img = frame_queue.get()
            yolo_result = analyze_image_with_yolo(img)

            key_action, mouse_action, key_value, key_log_prob, mouse_value, mouse_log_prob = agent.act(img)

            if key_action is None and mouse_action is None:
                print("Ação inválida, pulando passo.")
                continue

            keyboard_reward = agent.get_keyboard_reward(img, key_action, yolo_result) if TRAIN_KEYBOARD else None
            mouse_reward = agent.get_mouse_reward(img, mouse_action, yolo_result) if TRAIN_MOUSE else None
            total_reward += (keyboard_reward or 0) + (mouse_reward or 0)

            agent.store(
                img,
                (key_action, mouse_action),
                keyboard_reward,
                mouse_reward,
                key_log_prob,
                key_value,
                mouse_log_prob,
                mouse_value
            )

            agent.step(img, yolo_result)  # Executa ações reais (opcional, pode comentar em testes offline)

            # A cada 100 passos, treina e salva
            if steps_taken > 0 and steps_taken % 100 == 0:
                agent.update()
                agent.save()

            steps_taken += 1
            time.sleep(STEP_DELAY)

        print(f"[EP {episode+1}] Total reward: {total_reward:.2f} | Steps: {steps_taken}")

        # Treina no fim do episódio
        agent.update()
        agent.save()

if __name__ == "__main__":
    train()
