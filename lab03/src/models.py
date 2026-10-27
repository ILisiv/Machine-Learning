import torch
import torch.nn as nn


class MLP(nn.Module):

  def __init__(self):
    super().__init__()
    self.flatten = nn.Flatten()
    self.net = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

  def forward(self, x):
    x = self.flatten(x)
    return self.net(x)


class CNN(nn.Module):

  def __init__(self):
    super().__init__()
    self.features = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),  # 14x14 -> 7x7
    )
    self.flatten = nn.Flatten()
    self.classifier = nn.Linear(32 * 7 * 7, 10)

  def forward(self, x):
    x = self.features(x)
    x = self.flatten(x)
    return self.classifier(x)


class AttentionModel(nn.Module):

  def __init__(self):
    super().__init__()
    # Split 28x28 image into 16 patches of 7x7 using Conv2d (kernel=7, stride=7)
    # Projects each 7x7 patch to d_model = 64
    self.patch_embed = nn.Conv2d(1, 64, kernel_size=7, stride=7)
    self.pos_embed = nn.Parameter(torch.randn(1, 16, 64))

    encoder_layer = nn.TransformerEncoderLayer(
        d_model=64, nhead=4, dim_feedforward=128, batch_first=True
    )
    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
    self.fc = nn.Linear(64, 10)

  def forward(self, x):
    # x shape: [Batch, 1, 28, 28]
    x = self.patch_embed(x)  # Shape: [Batch, 64, 4, 4]
    x = x.flatten(2).transpose(1, 2)  # Shape: [Batch, 16, 64]
    x = x + self.pos_embed
    x = self.transformer(x)  # Shape: [Batch, 16, 64]
    x = x.mean(dim=1)  # Mean pooling across the 16 patches -> [Batch, 64]
    return self.fc(x)


# Quick verification of parameter counts
if __name__ == "__main__":
  models = {"MLP": MLP(), "CNN": CNN(), "Attention": AttentionModel()}
  for name, model in models.items():
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name} Model Parameters: {params:,}")