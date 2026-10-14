import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- 1. Reproducibility Seeds ---
random.seed(42)
np.random.seed(42)

# --- 2. Dataset Preparation ---
# Load optical recognition of handwritten digits, scale inputs by dividing by 16
X, y = load_digits(return_X_y=True)
X = X / 16.0

# One-hot encode labels for 10 classes
Y_one_hot = np.eye(10)[y]

# 70/30 stratified split with random_state=42
X_train, X_test, y_train_oh, y_test_oh, y_train_raw, y_test_raw = train_test_split(
    X, Y_one_hot, y, test_size=0.3, stratify=y, random_state=42
)

# --- 3. Neural Network Class from Scratch ---
class TwoLayerMLP:
    def __init__(self, input_dim=64, hidden_dim=64, output_dim=10):
        # Initialize weights from N(0, 0.01) and biases to zeros
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((hidden_dim,))
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b2 = np.zeros((hidden_dim,))
        self.W3 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b3 = np.zeros((output_dim,))

    def forward(self, X):
        # Layer 1 (Hidden 1) -> ReLU
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0, self.Z1)
        
        # Layer 2 (Hidden 2) -> ReLU
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = np.maximum(0, self.Z2)
        
        # Layer 3 (Output) -> Softmax
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        exp_scores = np.exp(self.Z3 - np.max(self.Z3, axis=1, keepdims=True))
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs

    def compute_loss(self, y_true_oh):
        m = y_true_oh.shape[0]
        loss = -np.sum(y_true_oh * np.log(self.probs + 1e-9)) / m
        return loss

    def backward(self, X, y_true_oh):
        m = X.shape[0]
        
        # Gradient of cross-entropy with softmax
        dZ3 = self.probs - y_true_oh
        
        # Gradients for layer 3
        self.dW3 = np.dot(self.A2.T, dZ3) / m
        self.db3 = np.sum(dZ3, axis=0) / m
        
        # Gradients for layer 2
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * (self.Z2 > 0)  # ReLU derivative
        self.dW2 = np.dot(self.A1.T, dZ2) / m
        self.db2 = np.sum(dZ2, axis=0) / m
        
        # Gradients for layer 1
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.Z1 > 0)  # ReLU derivative
        self.dW1 = np.dot(X.T, dZ1) / m
        self.db1 = np.sum(dZ1, axis=0) / m

# --- 4. Sanity Check: Untrained Loss ---
net_check = TwoLayerMLP(input_dim=64, hidden_dim=64, output_dim=10)
initial_probs = net_check.forward(X_train)
initial_loss = net_check.compute_loss(y_train_oh)
expected_loss = np.log(10)
print(f"Sanity Check -> Initial Loss: {initial_loss:.4f} (Expected ~ln(10) = {expected_loss:.4f})")

# --- 5. Gradient Checking Function ---
def gradient_check(net, X_sample, y_sample_oh):
    net.forward(X_sample)
    net.backward(X_sample, y_sample_oh)
    
    analytic_grads = [net.dW1, net.db1, net.dW2, net.db2, net.dW3, net.db3]
    params = [net.W1, net.b1, net.W2, net.b2, net.W3, net.b3]
    
    eps = 1e-5
    passed = 0
    total = 10
    
    for _ in range(total):
        p_idx = random.randint(0, len(params) - 1)
        p = params[p_idx]
        grad = analytic_grads[p_idx]
        
        idx = tuple(np.random.randint(0, dim) for dim in p.shape)
        orig_val = p[idx]
        
        p[idx] = orig_val + eps
        net.forward(X_sample)
        loss_plus = net.compute_loss(y_sample_oh)
        
        p[idx] = orig_val - eps
        net.forward(X_sample)
        loss_minus = net.compute_loss(y_sample_oh)
        
        p[idx] = orig_val
        
        num_grad = (loss_plus - loss_minus) / (2 * eps)
        an_grad = grad[idx]
        
        numerator = np.abs(num_grad - an_grad)
        denominator = max(1e-9, np.abs(num_grad) + np.abs(an_grad))
        rel_error = numerator / denominator
        
        if rel_error < 1e-6:
            passed += 1

    print(f"Gradient Check: {passed}/{total} weights passed with relative error < 1e-6")
    return passed >= 9

sample_idx = np.random.choice(X_train.shape[0], 5, replace=False)
gradient_check(net_check, X_train[sample_idx], y_train_oh[sample_idx])

# --- 6. Training Loop & Hyperparameter Sweeps ---
learning_rates = [0.001, 0.01, 0.1]
widths = [16, 64]
batch_size = 32
epochs = 50

os.makedirs("lab02/results", exist_ok=True)
plt.figure(figsize=(12, 8))

results_summary = []

for W in widths:
    for lr in learning_rates:
        print(f"Training MLP with Width={W}, Learning Rate={lr}...")
        net = TwoLayerMLP(input_dim=64, hidden_dim=W, output_dim=10)
        loss_history = []
        
        for epoch in range(epochs):
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train_oh[indices]
            
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                net.forward(X_batch)
                net.backward(X_batch, y_batch)
                
                net.W1 -= lr * net.dW1
                net.b1 -= lr * net.db1
                net.W2 -= lr * net.dW2
                net.b2 -= lr * net.db2
                net.W3 -= lr * net.dW3
                net.b3 -= lr * net.db3
            
            net.forward(X_train)
            epoch_loss = net.compute_loss(y_train_oh)
            loss_history.append(epoch_loss)
            
        test_probs = net.forward(X_test)
        test_preds = np.argmax(test_probs, axis=1)
        test_acc = accuracy_score(y_test_raw, test_preds)
        
        results_summary.append({"Width": W, "LR": lr, "Test Accuracy": test_acc})
        plt.plot(loss_history, label=f"W={W}, lr={lr}")

plt.title("Training Loss Curves Across Hyperparameters")
plt.xlabel("Epochs")
plt.ylabel("Cross-Entropy Loss")
plt.legend()
plt.grid(True)
plt.savefig("lab02/results/loss_curves.png")
plt.close()
print("Saved loss curves to lab02/results/loss_curves.png")

# --- 7. Execution Timing Breakdown (Median of 100 Runs) ---
net_profile = TwoLayerMLP(input_dim=64, hidden_dim=64, output_dim=10)
sample_batch = X_train[0:32]
sample_y = y_train_oh[0:32]

forward_times = []
backward_times = []

for _ in range(100):
    start = time.perf_counter()
    net_profile.forward(sample_batch)
    end = time.perf_counter()
    forward_times.append((end - start) * 1000)

    start = time.perf_counter()
    net_profile.backward(sample_batch, sample_y)
    end = time.perf_counter()
    backward_times.append((end - start) * 1000)

median_fwd = np.median(forward_times)
median_bwd = np.median(backward_times)

print("\nTiming Breakdown (Median of 100 runs):")
print(f"Forward Pass: {median_fwd:.4f} ms")
print(f"Backward Pass: {median_bwd:.4f} ms")

print("\nHyperparameter Sweep Summary (Test Accuracies):")
for res in results_summary:
    print(f"Width: {res['Width']}, LR: {res['LR']} -> Accuracy: {res['Test Accuracy']:.4f}")