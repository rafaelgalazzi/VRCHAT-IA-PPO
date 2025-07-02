# Script to train PPO agent
import time
from agents.ppo_agent import VRChatAgent
from utils.screen_utils import capture_vrchat_frame
from utils.yolo_utils import load_yolo_model, analyze_image_with_yolo

print("[INFO] Iniciando treino PPO com teclado e mouse")
agent = VRChatAgent()
yolo = load_yolo_model()

EPISODES = 5
STEPS_PER_EPISODE = 300

for ep in range(EPISODES):
    print(f"[EPISODE {ep+1}] Iniciando...")
    total_reward = 0
    for step in range(STEPS_PER_EPISODE):
        img = capture_vrchat_frame()
        if img is None:
            continue
        yolo_result = analyze_image_with_yolo(yolo, img)

        key_action, mouse_action = agent.act(img)
        agent.apply_actions(key_action, mouse_action)
        reward = 1.0  # substitua pela lógica de recompensa
        total_reward += reward
        time.sleep(0.05)

    print(f"[EPISODE {ep+1}] Total reward: {total_reward:.2f}")
    # Aqui você pode incluir agent.update() e agent.save() se desejar
