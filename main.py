import os
import subprocess

OPTIONS = {
    "1": ("Gravar dados para imitação", "python data/record_screen.py"),
    "2": ("Treinar modelo de imitação", "python imitation/train_imitation.py"),
    "3": ("Testar modelo de imitação", "python imitation/test_imitation.py"),
    "4": ("Treinar com PPO", "python scripts/train_ppo.py"),
    "5": ("Rodar IA com modelo treinado (inferência)", None),  # Tratado manualmente
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

def main():
    while True:
        print("\n== MENU PRINCIPAL ==")
        for k, (desc, _) in OPTIONS.items():
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
        else:
            print(f"\n[INFO] Executando: {desc}\n")
            subprocess.run(command, shell=True)

if __name__ == "__main__":
    main()
