import random
import time
import os
import psutil
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Set global seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 2. Load dataset and split 70/30 stratified
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []
process = psutil.Process(os.getpid())

for name, model in models.items():
    print(f"\nEvaluating {name}...")

    # --- Training Time (1 warm-up, then 5 runs, report median) ---
    train_times = []
    # Warm-up run
    model.fit(X_train, y_train)
    
    for _ in range(5):
        start_mem = process.memory_info().rss
        start_time = time.perf_counter()
        
        model.fit(X_train, y_train)
        
        end_time = time.perf_counter()
        end_mem = process.memory_info().rss
        train_times.append(end_time - start_time)
    
    median_train_time = np.median(train_times)
    peak_mem_train = end_mem  # RSS peak approximation

    # --- Single-Sample Inference Latency (repeat 100 times, report median) ---
    single_sample = X_test[0:1]
    latencies = []
    for _ in range(100):
        start_time = time.perf_counter()
        _ = model.predict(single_sample)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)  # in milliseconds
    
    median_latency = np.median(latencies)

    # --- Accuracy ---
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # --- Model Size ---
    model_path = f"lab01/results/{name.lower()}_model.joblib"
    joblib.dump(model, model_path)
    file_size_bytes = os.path.getsize(model_path)
    file_size_kb = file_size_bytes / 1024

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Median Train Time (s)": round(median_train_time, 4),
        "Median Inference Latency (ms)": round(median_latency, 4),
        "Model Size (Bytes)": file_size_bytes,
        "Model Size (KB)": round(file_size_kb, 2),
        "Peak RSS Memory (MB)": round(peak_mem_train / (1024 * 1024), 2)
    })

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv("lab01/results/baseline_accuracy.csv", index=False)
print("\nMeasurements completed successfully!")
print(df_results)