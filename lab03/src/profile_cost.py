import time
import torch
from models import CNN, AttentionModel, MLP
from thop import profile
from torchinfo import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def benchmark_model(model_name, model, input_size):
  print(f"\n{'='*20} {model_name} {'='*20}")
  model = model.to(device)
  model.eval()

  # 1. torchinfo summary (Parameters & Shapes)
  print("\n--- torchinfo Summary ---")
  summary(model, input_size=input_size, device=device)

  # 2. thop profile (FLOPs & Parameters)
  print("\n--- thop Profile ---")
  dummy_input = torch.randn(input_size).to(device)
  macs, params = profile(model, inputs=(dummy_input,), verbose=False)
  print(
      f"thop FLOPs (approx 2 * MACs): {2 * macs:,.0f} | thop Parameters:"
      f" {params:,}"
  )

  # 3. Latency Measurement
  print("\n--- Latency Benchmark ---")
  for batch_size, runs in [(1, 100), (128, 20)]:
    b_input = torch.randn(batch_size, *input_size[1:]).to(device)

    # Warmup runs
    with torch.no_grad():
      for _ in range(10):
        _ = model(b_input)
    if device.type == "cuda":
      torch.cuda.synchronize()

    # Timed runs
    latencies = []
    with torch.no_grad():
      for _ in range(runs):
        start = time.perf_counter()
        _ = model(b_input)
        if device.type == "cuda":
          torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    latencies.sort()
    median_latency = latencies[len(latencies) // 2]
    print(
        f"Batch size {batch_size:3d} ({runs:3d} runs) -> Median Latency:"
        f" {median_latency:.3f} ms"
    )


if __name__ == "__main__":
  # Input shape: [Batch, Channels, Height, Width]
  input_size = (1, 1, 28, 28)

  benchmark_model("MLP", MLP(), input_size)
  benchmark_model("CNN", CNN(), input_size)
  benchmark_model("Attention Model", AttentionModel(), input_size)