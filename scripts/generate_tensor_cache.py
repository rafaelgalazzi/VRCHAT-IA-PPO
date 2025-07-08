import os
import csv
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Caminhos
IMAGE_DIR = "data/images"
LABEL_FILE = "data/labels.csv"
CACHE_DIR = "data/image_cache"

os.makedirs(CACHE_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

with open(LABEL_FILE, "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

seen = set()
for row in tqdm(rows, desc="Generating image cache .jpeg"):
    img_name = row["image"]

    if not img_name.lower().endswith(".jpeg"):
        continue

    if img_name in seen:
        continue
    seen.add(img_name)

    img_path = os.path.join(IMAGE_DIR, img_name)
    base_name = os.path.splitext(img_name)[0]
    cache_path = os.path.join(CACHE_DIR, base_name + ".pt")

    if os.path.exists(cache_path):
        continue 

    try:
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img)
        torch.save(tensor, cache_path)
    except Exception as e:
        print(f"[ERROR] Processing {img_name}: {e}")
