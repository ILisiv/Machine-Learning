import time
import torch
from torchinfo import summary
from thop import profile
from models import FashionMLP, FashionCNN, FashionAttention

def analyze_costs():
    models = {
        "MLP": FashionMLP(),
        "CNN": FashionCNN(),
        "Attention": FashionAttention()
    }

    dummy_input_b1 = torch.randn(1, 1, 28, 28)
    dummy_input_b128 = torch.randn(128, 1, 28, 28)

    for name, model in models.items() :
        print(f"\n--- Cost Analysis for {name} ---")
        
        # 1. Parameter count via torchinfo
        model_summary = summary(model, input_size=(1, 1, 28, 28), verbose=0)
        print(f"Total Parameters: {model_summary.total_params}")

        # 2. FLOPs via thop
        macs, params = profile(model, inputs=(dummy_input_b1,), verbose=False)
        flops = macs * 2  # MACs to FLOPs conversion
        print(f"Estimated FLOPs (thop): {flops}")

        # 3. Inference Latency (Batch Size 1 - Median of 100 runs)
        latencies_b1 = []
        model.eval()
        with torch.no_grad():
            for _ in range(100):
                start = time.time()
                _ = model(dummy_input_b1)
                latencies_b1.append((time.time() - start) * 1000) # ms
        latencies_b1.sort()
        print(f"Latency Batch 1 (Median): {latencies_b1[50]:.2f} ms")

        # 4. Inference Latency (Batch Size 128 - Median of 20 runs)
        latencies_b128 = []
        with torch.no_grad():
            for _ in range(20):
                start = time.time()
                _ = model(dummy_input_b128)
                latencies_b128.append((time.time() - start) * 1000) # ms
        latencies_b128.sort()
        print(f"Latency Batch 128 (Median): {latencies_b128[10]:.2f} ms")

if __name__ == "__main__":
    analyze_costs()