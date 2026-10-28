import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from models import FashionMLP, FashionCNN, FashionAttention

def train_model(model, train_loader, test_loader, epochs=5, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    # Evaluate on Test Set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)
    
    models = {
        "MLP": FashionMLP(),
        "CNN": FashionCNN(),
        "Attention": FashionAttention()
    }

    for name, model in models.items():
        print(f"Training {name} for 5 epochs...")
        acc = train_model(model, train_loader, test_loader, epochs=5, lr=1e-3)
        print(f"[+] {name} Test Accuracy: {acc * 100:.2f}%\n")