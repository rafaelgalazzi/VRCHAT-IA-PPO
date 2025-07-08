import os
import subprocess

# Important paths
CHECKPOINT_IMITATION = "checkpoint_latest.pth"

OPTIONS = {
    "0": ("Generate image cache (.pt)", "python scripts/generate_tensor_cache.py"),
    "1": ("Record data for imitation", "python data/record_screen.py"),
    "2": ("Train imitation model", "python imitation/train_imitation.py"),
    "3": ("Test imitation model", "python imitation/test_imitation.py"),
    "4": ("Train with PPO", "python scripts/train_ppo.py"),
    "5": ("Run AI with trained model (inference)", None),
    "q": ("Quit", None)
}

def run_inference():
    print("\n[INFO] Choose the model to run:")
    print("[1] Imitation (supervised model)")
    print("[2] PPO (reinforcement learning model)")

    choice = input("Model: ").strip()
    if choice == "1":
        command = "python scripts/run_imitation.py"
    elif choice == "2":
        command = "python scripts/run_inference.py"
    else:
        print("Invalid option.")
        return

    print(f"\n[INFO] Running: {command}")
    subprocess.run(command, shell=True)

def run_with_options(command: str, checkpoint_path: str):
    # Ask whether to use image cache
    use_cache = input("Use image cache (.pt)? (y/n): ").strip().lower()
    os.environ["USE_IMAGE_CACHE"] = "1" if use_cache == "y" else "0"

    # Check for existing checkpoint
    if os.path.exists(checkpoint_path):
        print(f"\n[INFO] Checkpoint found at '{checkpoint_path}'")
        resume = input("Resume from last checkpoint? (y/n): ").strip().lower()
        if resume == "y":
            os.environ["RESUME_CHECKPOINT"] = "1"
            print("[INFO] Resuming training...")

    print(f"\n[INFO] Running: {command}\n")
    subprocess.run(command, shell=True)

def main():
    while True:
        print("\n== MAIN MENU ==")
        for k, (desc, _) in sorted(OPTIONS.items()):
            print(f"[{k}] {desc}")
        
        choice = input("Choose an option: ").lower()
        if choice not in OPTIONS:
            print("Invalid option. Please try again.")
            continue

        desc, command = OPTIONS[choice]
        if choice == "q":
            print("Exiting...")
            break

        if choice == "5":
            run_inference()
        elif choice == "2":
            run_with_options(command, CHECKPOINT_IMITATION)
        else:
            print(f"\n[INFO] Running: {desc}\n")
            subprocess.run(command, shell=True)

if __name__ == "__main__":
    main()
