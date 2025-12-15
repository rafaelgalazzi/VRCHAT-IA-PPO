import os
import subprocess

# Important paths
CHECKPOINT_IMITATION = "checkpoint_latest.pth"

OPTIONS = {
    "0": ("Generate image cache (.pt)", "python scripts/generate_tensor_cache.py"),
    "1": ("Record data for imitation", "python data/record_screen.py"),
    "2": ("Train imitation model", "python imitation/train_imitation.py"),
    "3": ("Test imitation model", "python imitation/test_imitation.py"),
    "4": ("Run imitation model", "python scripts/run_imitation.py"),
    "q": ("Quit", None)
}

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

        if choice == "2":
            run_with_options(command, CHECKPOINT_IMITATION)
        else:
            print(f"\n[INFO] Running: {desc}\n")
            subprocess.run(command, shell=True)

if __name__ == "__main__":
    main()
