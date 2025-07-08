# VRChat AI – Imitation and Reinforcement Learning Agent

This project is an AI system designed to interact with VRChat in real-time. It uses two learning approaches: imitation learning (where it learns from example behavior) and reinforcement learning (where it learns by trial and error using rewards).

The agent captures what's happening on screen, analyzes it with a YOLOv8 object detector, and then decides which keyboard and mouse actions to perform. It's a hands-on experiment in applying computer vision and deep reinforcement learning to a real-world-like environment.

---

## What it can do

- Capture real-time gameplay from the VRChat window
- Use YOLOv8 to detect and understand objects in the environment
- Train an agent using:
  - Imitation learning (supervised learning from recorded data)
  - PPO (Proximal Policy Optimization, a reinforcement learning algorithm)
- Control both keyboard and mouse actions
- Normalize and apply mouse movement actions smoothly
- Save and resume training from checkpoints
- Run inference using a trained model

---

## Project structure

```
<pre> project-root/ ├── scripts/ │ ├── generate_tensor_cache.py ← Converts images to tensor files │ ├── train_ppo.py ← Trains PPO model │ ├── run_imitation.py ← Runs imitation model │ ├── run_inference.py ← Runs PPO model ├── imitation/ │ ├── train_imitation.py ← Trains supervised imitation model │ ├── test_imitation.py ← Tests imitation model ├── data/ │ ├── record_screen.py ← Records gameplay for imitation │ ├── mouse_normalization.json ← Mouse movement scaling data ├── agents/ │ └── ppo_agent.py ← PPO agent logic ├── utils/ │ └── yolo_utils.py ← YOLOv8 object detection ├── main_menu.py ← CLI to access all scripts ├── globals.py ← Training loop and capture logic └── README.md ← Project documentation </pre>
```

---

## How to use

### Imitation learning

1. Record gameplay data (screens + actions):
```bash
python data/record_screen.py
```

2. Train the imitation model:
```bash
python imitation/train_imitation.py
```

3. Test the model:
```bash
python imitation/test_imitation.py
```

4. Run inference:
```bash
python scripts/run_imitation.py
```

### PPO (Reinforcement Learning)

1. Train the PPO model:
```bash
python scripts/train_ppo.py
```

2. Run inference with the trained model:
```bash
python scripts/run_inference.py
```

---

## Interactive menu

You can use a CLI menu to access everything in one place:

```bash
python main_menu.py
```

---

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- mss
- Pillow
- keyboard
- psutil
- YOLOv8 (via `ultralytics`)
- A CUDA-compatible GPU (recommended for training)

To install everything:
```bash
pip install -r requirements.txt
```

---

## Notes

- VRChat must be running in **windowed mode** so it can be captured correctly.
- YOLOv8 is used for object detection. Make sure the `yolov8n.pt` model file is available.
- During PPO training, progress is automatically saved and can be resumed if interrupted.

---

## License

This project is open source and licensed under the MIT License.

---

## Acknowledgments

- YOLOv8 by Ultralytics
- PPO concepts inspired by OpenAI's work
- Thanks to the open-source community for tools and frameworks that made this possible
