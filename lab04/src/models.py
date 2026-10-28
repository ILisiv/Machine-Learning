import torch
import torch.nn as nn

# 1. MLP Architecture
class FashionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

# 2. CNN Architecture
class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        return self.fc(x)

# 3. Attention (Vision Transformer style) Architecture
class FashionAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # 28x28 image -> 16 patches of 7x7. Patch dim = 7 * 7 = 49
        self.patch_dim = 49
        self.projection = nn.Linear(49, 64) # Linear projection to d_model=64
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        B, C, H, W = x.shape  # B, 1, 28, 28
        # Reshape into 16 patches of 7x7
        x = x.unfold(2, 7, 7).unfold(3, 7, 7) # B, 1, 4, 4, 7, 7
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, 16, -1) # B, 16, 49
        
        x = self.projection(x) # B, 16, 64
        x = self.transformer(x) # B, 16, 64
        x = x.mean(dim=1) # Mean pooling over sequence dimension -> B, 64
        return self.fc(x) # B, 10