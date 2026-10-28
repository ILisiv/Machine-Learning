import matplotlib.pyplot as plt
import os

def generate_plots():
    models = ['MLP', 'CNN', 'Attention']
    parameters = [235146, 20490, 70794]
    latencies_b1 = [0.00, 0.00, 1.00]  # in ms
    latencies_b128 = [0.00, 8.00, 7.00] # in ms
    accuracies = [87.51, 89.40, 83.41]  # in %

    os.makedirs("results", exist_ok=True)

    # Plot 1: Accuracy vs Parameters
    plt.figure(figsize=(8, 5))
    plt.scatter(parameters, accuracies, color=['blue', 'green', 'orange'], s=100)
    for i, txt in enumerate(models):
        plt.annotate(txt, (parameters[i], accuracies[i]), fontsize=12, xytext=(5, 5), textcoords='offset points')
    plt.xlabel("Number of Parameters")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy vs. Model Parameters")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("results/accuracy_vs_parameters.png")
    plt.close()

    # Plot 2: Accuracy vs Latency (Batch 128)
    plt.figure(figsize=(8, 5))
    plt.scatter(latencies_b128, accuracies, color=['blue', 'green', 'orange'], s=100)
    for i, txt in enumerate(models):
        plt.annotate(txt, (latencies_b128[i], accuracies[i]), fontsize=12, xytext=(5, 5), textcoords='offset points')
    plt.xlabel("Inference Latency Batch 128 (ms)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy vs. Inference Latency (Batch 128)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("results/accuracy_vs_latency.png")
    plt.close()

    print("[+] Plots successfully generated and saved to results/ folder!")

if __name__ == "__main__":
    generate_plots()