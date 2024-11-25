import pandas as pd
import matplotlib.pyplot as plt

# Ler o CSV (substitua pelo caminho correto do seu arquivo)
data = pd.read_csv("model_results.csv")

# Gráficos para Visualização
def plot_metrics(data):
    # Filtrando pelo Batch Size
    batch_sizes = data["Batch Size"].unique()
    plt.figure(figsize=(14, 10))
    
    # Plotando Train Accuracy e Validation Accuracy
    for batch in batch_sizes:
        subset = data[data["Batch Size"] == batch]
        plt.plot(subset["Epochs"], subset["Train Accuracy"], marker='o', label=f"Train Acc (Batch={batch})")
        plt.plot(subset["Epochs"], subset["Validation Accuracy"], marker='x', linestyle='--', label=f"Val Acc (Batch={batch})")
    
    plt.title("Train vs Validation Accuracy por Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotando Train Loss e Validation Loss
    plt.figure(figsize=(14, 10))
    for batch in batch_sizes:
        subset = data[data["Batch Size"] == batch]
        plt.plot(subset["Epochs"], subset["Train Loss"], marker='o', label=f"Train Loss (Batch={batch})")
        plt.plot(subset["Epochs"], subset["Validation Loss"], marker='x', linestyle='--', label=f"Val Loss (Batch={batch})")
    
    plt.title("Train vs Validation Loss por Épocas")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# Chamando a função para plotar
plot_metrics(data)
