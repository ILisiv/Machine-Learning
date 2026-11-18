# src/train.py
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

class PyTorchMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def benchmark_execution():
    # Ensure lab05/results directory exists
    os.makedirs("results", exist_ok=True)

    X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    criterion = nn.BCELoss()

    # 1. Eager Mode Timings (Median of 5 runs)
    print("Running Eager Mode benchmarks...")
    eager_times = []
    for _ in range(5):
        model_eager = PyTorchMLP()
        opt = torch.optim.SGD(model_eager.parameters(), lr=0.1)
        start = time.perf_counter()
        for epoch in range(50):
            opt.zero_grad()
            out = model_eager(X_t)
            loss = criterion(out, y_t)
            loss.backward()
            opt.step()
        eager_times.append(time.perf_counter() - start)
    median_eager = np.median(eager_times)

    # 2. Traced / Compiled Mode Timings (Using torch.jit.trace to avoid Windows C++ compiler errors)
    print("Running Traced/Compiled Mode benchmarks...")
    compiled_times = []
    compile_mode_used = "torch.jit.trace"
    for _ in range(5):
        model_comp = PyTorchMLP()
        model_comp.eval() # set to eval for tracing
        example_input = torch.randn(1, 2)
        try:
            compiled_model = torch.jit.trace(model_comp, example_input)
        except Exception:
            compiled_model = model_comp # Fallback to standard if trace fails
            compile_mode_used = "eager-fallback"
        
        compiled_model.train()
        opt = torch.optim.SGD(model_comp.parameters(), lr=0.1)
        start = time.perf_counter()
        for epoch in range(50):
            opt.zero_grad()
            out = compiled_model(X_t)
            loss = criterion(out, y_t)
            loss.backward()
            opt.step()
        compiled_times.append(time.perf_counter() - start)
    median_compiled = np.median(compiled_times)

    # 3. Non-contiguous memory impact test
    print("Running tensor contiguity benchmarks...")
    x_contig = torch.randn(1000, 1000)
    x_non_contig = x_contig.t()  # non-contiguous view
    
    t0 = time.perf_counter()
    for _ in range(1000):
        _ = x_contig @ x_contig
    time_contig = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(1000):
        _ = x_non_contig @ x_non_contig
    time_non_contig = time.perf_counter() - t0

    # Print results to terminal
    print("\n--- Benchmark Results ---")
    print(f"Eager Execution Median Time (5 runs):    {median_eager:.4f}s")
    print(f"Compiled Execution ({compile_mode_used}): {median_compiled:.4f}s")
    print(f"Contiguous MatMul Time:                  {time_contig:.4f}s")
    print(f"Non-Contiguous MatMul Time:              {time_non_contig:.4f}s")

    # Save results to lab05/results/benchmark_metrics.json
    results_data = {
        "eager_median_seconds": float(median_eager),
        "compiled_median_seconds": float(median_compiled),
        "compile_mode": compile_mode_used,
        "contiguous_matmul_seconds": float(time_contig),
        "non_contiguous_matmul_seconds": float(time_non_contig)
    }

    results_path = os.path.join("results", "benchmark_metrics.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=4)
    print(f"\nSuccessfully saved metrics to {results_path}")

if __name__ == "__main__":
    benchmark_execution()