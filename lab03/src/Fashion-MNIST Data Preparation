import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# 1. Define the required transformations
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
)

# 2. Download and load the official Fashion-MNIST training dataset (60,000 samples)
full_train_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)

# 3. Split the 60k training set into 55k training and 5k validation sets
# Using a manual seed ensures reproducibility across runs
train_size = 55000
val_size = 5000
train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)

# 4. Load the official test dataset (10,000 samples)
test_dataset = datasets.FashionMNIST(
    root="./data", train=False, download=True, transform=transform
)

# 5. Create DataLoaders with a batch size of 128
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Verification prints
print(f"Train batches: {len(train_loader)} ({len(train_dataset)} samples)")
print(f"Validation batches: {len(val_loader)} ({len(val_dataset)} samples)")
print(f"Test batches: {len(test_loader)} ({len(test_dataset)} samples)")
print(f"Batch size: {batch_size}")