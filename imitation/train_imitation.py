import os
import csv
import json
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
from PIL import Image
from torchvision import transforms

# Imports internos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agents.imitation_agent import ImitationAgentLSTM
from utils.input_controller import TRAINING_KEYS, MOUSE_BUTTONS

# Configurações
IMAGE_DIR = "data/images"
CACHE_DIR = "data/image_cache"
LABEL_FILE = "data/labels.csv"
NORM_FILE = "data/mouse_normalization.json"
CHECKPOINT_FILE = "checkpoint_latest.pth"
EPOCHS = 20
BATCH_SIZE = 16
SEQ_LEN = 6
FRAME_DELAY = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_IMAGE_CACHE = os.environ.get("USE_IMAGE_CACHE", "1") == "1"

def print_memory_usage(note=""):
    mem = psutil.virtual_memory()
    print(f"[RAM] {note} - Uso: {mem.used // (1024 ** 2)} MB / {mem.total // (1024 ** 2)} MB")

class WindowDataset(Dataset):
    def __init__(self, label_file, use_cache=True, cache_dir=None, seq_len=6, frame_delay=5, norm_file=NORM_FILE):
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.seq_len = seq_len
        self.frame_delay = frame_delay

        with open(label_file, "r") as f:
            reader = csv.DictReader(f)
            self.raw_data = list(reader)

        self.valid_indices = []
        for i in range(len(self.raw_data) - frame_delay - (seq_len - 1)):
            session = self.raw_data[i]["session_id"]
            valid = all(self.raw_data[i + j]["session_id"] == session for j in range(seq_len + frame_delay))
            if valid:
                self.valid_indices.append(i)

        dx_vals, dy_vals = [], []
        for i in self.valid_indices:
            dx = float(self.raw_data[i + frame_delay]["mouse_dx"])
            dy = float(self.raw_data[i + frame_delay]["mouse_dy"])
            dx_vals.append(dx)
            dy_vals.append(dy)

        self.max_dx = max(1.0, max(abs(x) for x in dx_vals))
        self.max_dy = max(1.0, max(abs(y) for y in dy_vals))

        with open(norm_file, "w") as f:
            json.dump({"max_dx": self.max_dx, "max_dy": self.max_dy}, f, indent=4)
        print(f"[INFO] Mouse normalization saved to '{norm_file}'")
        print_memory_usage("After dataset init")

        if not self.use_cache:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        base_idx = self.valid_indices[idx]
        frames = []

        for j in range(self.seq_len):
            img_file = self.raw_data[base_idx + j]["image"]
            base_name = os.path.splitext(img_file)[0]

            if self.use_cache:
                path = os.path.join(self.cache_dir, base_name + ".pt")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"[ERROR] Missing cached file: {path}")
                tensor = torch.load(path)
            else:
                path = os.path.join(IMAGE_DIR, img_file)
                image = Image.open(path).convert("RGB")
                tensor = self.transform(image)

            frames.append(tensor)

        frame_sequence = torch.stack(frames, dim=0)

        target = self.raw_data[base_idx + self.frame_delay]
        keys = target["keys"].split("+") if target["keys"] else []
        key_vec = [int(k in keys) for k in TRAINING_KEYS]  # Fixed training keys
        dx = float(target["mouse_dx"]) / self.max_dx
        dy = float(target["mouse_dy"]) / self.max_dy

        # Handle mouse button clicks
        mouse_buttons = target["mouse_buttons"].split("+") if target["mouse_buttons"] else []
        mouse_button_vec = [int(b in mouse_buttons) for b in MOUSE_BUTTONS]

        return frame_sequence, torch.tensor(key_vec, dtype=torch.float32), torch.tensor([dx, dy], dtype=torch.float32), torch.tensor(mouse_button_vec, dtype=torch.float32)

def print_progress_bar(iteration, total, prefix='', suffix='', length=30):
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '#' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()

if __name__ == "__main__":
    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Using image cache: {USE_IMAGE_CACHE}")

    dataset = WindowDataset(
        label_file=LABEL_FILE,
        use_cache=USE_IMAGE_CACHE,
        cache_dir=CACHE_DIR,
        seq_len=SEQ_LEN,
        frame_delay=FRAME_DELAY
    )

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

    keyboard_model = ImitationAgentLSTM(output_dim=len(TRAINING_KEYS), mode="keyboard").to(DEVICE)
    mouse_model = ImitationAgentLSTM(output_dim=2, mode="mouse").to(DEVICE)

    k_optimizer = optim.Adam(keyboard_model.parameters(), lr=1e-4)
    m_optimizer = optim.Adam(mouse_model.parameters(), lr=1e-4)
    bce = nn.BCELoss()
    mse = nn.MSELoss()

    start_epoch = 0
    if os.path.exists(CHECKPOINT_FILE):
        resume = os.environ.get("RESUME_CHECKPOINT", "0") == "1"
        if resume:
            print(f"[INFO] Resuming from checkpoint: {CHECKPOINT_FILE}")
            checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
            keyboard_model.load_state_dict(checkpoint["keyboard_model_state"])
            mouse_model.load_state_dict(checkpoint["mouse_model_state"])
            # Load mouse click model state if available (for backward compatibility)
            if "mouse_click_model_state" in checkpoint:
                mouse_click_model.load_state_dict(checkpoint["mouse_click_model_state"])
            k_optimizer.load_state_dict(checkpoint["k_optimizer_state"])
            m_optimizer.load_state_dict(checkpoint["m_optimizer_state"])
            # Load mouse click optimizer state if available (for backward compatibility)
            if "mc_optimizer_state" in checkpoint:
                mc_optimizer.load_state_dict(checkpoint["mc_optimizer_state"])
            start_epoch = checkpoint["epoch"]

    # Create model for mouse button clicks
    mouse_click_model = ImitationAgentLSTM(output_dim=len(MOUSE_BUTTONS), mode="mouse_click").to(DEVICE)
    mc_optimizer = optim.Adam(mouse_click_model.parameters(), lr=1e-4)

    for epoch in range(start_epoch, EPOCHS):
        total_k_loss, total_m_loss, total_mc_loss = 0.0, 0.0, 0.0
        num_batches = len(dataloader)

        for i, (seqs, key_labels, mouse_labels, mouse_click_labels) in enumerate(dataloader, 1):
            seqs = seqs.to(DEVICE)
            key_labels = key_labels.to(DEVICE)
            mouse_labels = mouse_labels.to(DEVICE)
            mouse_click_labels = mouse_click_labels.to(DEVICE)

            k_optimizer.zero_grad()
            pred_keys = keyboard_model(seqs)
            k_loss = bce(pred_keys, key_labels)
            k_loss.backward()
            k_optimizer.step()
            total_k_loss += k_loss.item()

            m_optimizer.zero_grad()
            pred_mouse = mouse_model(seqs)
            m_loss = mse(pred_mouse, mouse_labels)
            m_loss.backward()
            m_optimizer.step()
            total_m_loss += m_loss.item()

            mc_optimizer.zero_grad()
            pred_mouse_clicks = mouse_click_model(seqs)
            mc_loss = bce(pred_mouse_clicks, mouse_click_labels)  # Use BCE for binary classification
            mc_loss.backward()
            mc_optimizer.step()
            total_mc_loss += mc_loss.item()

            print_progress_bar(i, num_batches, prefix=f"Epoch {epoch+1}", suffix="Training", length=40)

        print(f"\n[Epoch {epoch+1}] Keyboard Loss: {total_k_loss / num_batches:.4f} | Mouse Loss: {total_m_loss / num_batches:.4f} | Mouse Click Loss: {total_mc_loss / num_batches:.4f}")
        print_memory_usage(f"After epoch {epoch+1}")

        torch.save({
            "epoch": epoch + 1,
            "keyboard_model_state": keyboard_model.state_dict(),
            "mouse_model_state": mouse_model.state_dict(),
            "mouse_click_model_state": mouse_click_model.state_dict(),
            "k_optimizer_state": k_optimizer.state_dict(),
            "m_optimizer_state": m_optimizer.state_dict(),
            "mc_optimizer_state": mc_optimizer.state_dict()
        }, CHECKPOINT_FILE)

    torch.save(keyboard_model.state_dict(), "imitation_keyboard_latest.pth")
    torch.save(mouse_model.state_dict(), "imitation_mouse_latest.pth")
    torch.save(mouse_click_model.state_dict(), "imitation_mouse_click_latest.pth")
    print("✅ Final models saved.")
