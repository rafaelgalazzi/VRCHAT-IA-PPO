import os
import subprocess

# Caminhos importantes
CHECKPOINT_IMITATION = "checkpoint_latest.pth"

OPTIONS = {
    "0": ("Gerar cache de imagens (.pt)", "python scripts/generate_tensor_cache.py"),
    "1": ("Gravar dados para imitação", "python data/record_screen.py"),
    "2": ("Treinar modelo de imitação", "python imitation/train_imitation.py"),
    "3": ("Testar modelo de imitação", "python imitation/test_imitation.py"),
    "4": ("Treinar com PPO", "python scripts/train_ppo.py"),
    "5": ("Rodar IA com modelo treinado (inferência)", None),
    "q": ("Sair", None)
}

def run_inference():
    print("\n[INFO] Escolha o modelo para rodar:")
    print("[1] Imitação (modelo supervisionado)")
    print("[2] PPO (modelo por reforço)")

    choice = input("Modelo: ").strip()
    if choice == "1":
        command = "python scripts/run_imitation.py"
    elif choice == "2":
        command = "python scripts/run_inference.py"
    else:
        print("Opção inválida.")
        return

    print(f"\n[INFO] Executando: {command}")
    subprocess.run(command, shell=True)

def run_with_options(command: str, checkpoint_path: str):
    # Escolher se vai usar cache
    use_cache = input("Usar cache de imagens (.pt)? (s/n): ").strip().lower()
    os.environ["USE_IMAGE_CACHE"] = "1" if use_cache == "s" else "0"

    # Verificar se há checkpoint
    if os.path.exists(checkpoint_path):
        print(f"\n[INFO] Checkpoint detectado em '{checkpoint_path}'")
        resume = input("Deseja continuar de onde parou? (s/n): ").strip().lower()
        if resume == "s":
            os.environ["RESUME_CHECKPOINT"] = "1"
            print("[INFO] Retomando o treinamento...")

    print(f"\n[INFO] Executando: {command}\n")
    subprocess.run(command, shell=True)

def main():
    while True:
        print("\n== MENU PRINCIPAL ==")
        for k, (desc, _) in sorted(OPTIONS.items()):
            print(f"[{k}] {desc}")
        
        choice = input("Escolha uma opção: ").lower()
        if choice not in OPTIONS:
            print("Opção inválida. Tente novamente.")
            continue

        desc, command = OPTIONS[choice]
        if choice == "q":
            print("Saindo...")
            break

        if choice == "5":
            run_inference()
        elif choice == "2":
            run_with_options(command, CHECKPOINT_IMITATION)
        else:
            print(f"\n[INFO] Executando: {desc}\n")
            subprocess.run(command, shell=True)

if __name__ == "__main__":
    main()
