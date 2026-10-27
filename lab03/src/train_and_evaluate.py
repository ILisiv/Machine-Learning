import os
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from models import CNN, AttentionModel, MLP
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data_loaders():
  transform = transforms.Compose(
      [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
  )
  full_train = datasets.FashionMNIST(
      root="./data", train=True, download=True, transform=transform
  )
  test_dataset = datasets.FashionMNIST(
      root="./data", train=False, download=True, transform=transform
  )

  train_dataset, val_dataset = random_split(
      full_train, [55000, 5000], generator=torch.Generator().manual_seed(42)
  )

  train_loader = DataLoader(
      train_dataset, batch_size=128, shuffle=True, num_workers=2
  )
  test_loader = DataLoader(
      test_dataset, batch_size=128, shuffle=False, num_workers=2
  )
  return train_loader, test_loader


def train_and_evaluate(model, train_loader, test_loader, epochs=5):
  model = model.to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=1e-3)

  print(
      f"Training {model.__class__.__name__} for {epochs} epochs on"
      f" {device}..."
  )
  for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
      images, labels = images.to(device), labels.to(device)
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {epoch_loss:.4f}")

  # Evaluate on test set
  model.eval()
  correct = 0
  total = 0
  with torch.no_grad():
    for images, labels in test_loader:
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  test_accuracy = 100 * correct / total
  print(f"Test Accuracy: {test_accuracy:.2f}%\n")
  return test_accuracy


def measure_latency(model, batch_size=128, runs=20):
  model.eval()
  dummy_input = torch.randn(batch_size, 1, 28, 28).to(device)

  # Warmup
  with torch.no_grad():
    for _ in range(10):
      _ = model(dummy_input)
  if device.type == "cuda":
    torch.cuda.synchronize()

  latencies = []
  with torch.no_grad():
    for _ in range(runs):
      start = time.perf_counter()
      _ = model(dummy_input)
      if device.type == "cuda":
        torch.cuda.synchronize()
      end = time.perf_counter()
      latencies.append((end - start) * 1000)  # ms

  latencies.sort()
  return latencies[len(latencies) // 2]


def main():
  os.makedirs("lab03/results", exist_ok=True)
  train_loader, test_loader = get_data_loaders()

  models_dict = {"MLP": MLP(), "CNN": CNN(), "Attention": AttentionModel()}

  results = {}

  for name, model in models_dict.items():
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    test_acc = train_and_evaluate(model, train_loader, test_loader, epochs=5)
    latency = measure_latency(model, batch_size=128, runs=20)

    results[name] = {
        "params": params,
        "accuracy": test_acc,
        "latency": latency,
    }
    print(
        f"[{name}] Params: {params:,} | Accuracy: {test_acc:.2f}% | Latency:"
        f" {latency:.2f} ms\n"
    )

  # Step 5: Plotting
  names = list(results.keys())
  accuracies = [results[n]["accuracy"] for n in names]
  params_list = [results[n]["params"] for n in names]
  latencies = [results[n]["latency"] for n in names]
  colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

  # Plot 1: Accuracy vs Parameters
  plt.figure(figsize=(7, 5))
  plt.scatter(params_list, accuracies, color=colors, s=100)
  for i, txt in enumerate(names):
    plt.annotate(
        txt,
        (params_list[i], accuracies[i]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontweight="bold",
    )
  plt.xlabel("Number of Parameters")
  plt.ylabel("Test Accuracy (%)")
  plt.title("Accuracy vs. Parameter Count")
  plt.grid(True, linestyle="--", alpha=0.6)
  plt.savefig("lab03/results/accuracy_vs_params.png", bbox_inches="tight")
  plt.close()

  # Plot 2: Accuracy vs Latency
  plt.figure(figsize=(7, 5))
  plt.scatter(latencies, accuracies, color=colors, s=100)
  for i, txt in enumerate(names):
    plt.annotate(
        txt,
        (latencies[i], accuracies[i]),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontweight="bold",
    )
  plt.xlabel("Inference Latency (Batch 128, ms)")
  plt.ylabel("Test Accuracy (%)")
  plt.title("Accuracy vs. Inference Latency")
  plt.grid(True, linestyle="--", alpha=0.6)
  plt.savefig("lab03/results/accuracy_vs_latency.png", bbox_inches="tight")
  plt.close()

  print("Plots successfully saved in lab03/results/!")


if __name__ == "__main__":
  main()