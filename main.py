import os
import subprocess

OPTIONS = {
    "1": ("Gravar dados para imitação", "python data/record_screen.py"),
    "2": ("Treinar modelo de imitação", "python imitation/train_imitation.py"),
    "3": ("Testar modelo de imitação", "python imitation/test_imitation.py"),
    "4": ("Treinar com PPO", "python scripts/train_ppo.py"),
    "5": ("Rodar IA com modelo treinado (inferência)", "python scripts/run_inference.py"),
    "q": ("Sair", None)
}

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

        print(f"\n[INFO] Executando: {desc}\n")
        subprocess.run(command, shell=True)

if __name__ == "__main__":
    main()
