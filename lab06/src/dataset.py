import torch
from torch.utils.data import DataLoader
from sklearn.datasets import make_classification

def get_dataloader(batch_size=64, num_workers=2, n_samples=2000):
    X, y = make_classification(n_samples=n_samples, n_features=20, n_classes=2, random_state=42)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader