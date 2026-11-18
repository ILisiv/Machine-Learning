# Lab 6 Report: Training Pipelines, Profiling, and Project Proposal

## 1. YAML-Driven Training Pipeline
We restructured our model training into a modular pipeline controlled via `configs/train_config.yaml`. MLflow tracked hyperparameters, training curves, and checkpoints automatically.

## 2. Profiling & Bottleneck Analysis (`torch.profiler`)
Profiling revealed that execution time is heavily dominated by forward/backward compute passes when batch sizes are optimal, whereas smaller batch sizes suffer from data-loading queue overhead.

## 3. Worker Count & Batch Size Studies
* **`num_workers` Study:** Varying workers from $0$ to $4$ demonstrated optimal I/O throughput at `num_workers = 2`, preventing multiprocessing queue contention.
* **Batch-Size Study:** Testing batch sizes $[32, 64, 128, 256, 512]$ showed that throughput scales up to batch size $256$ before memory bandwidth saturation occurs.

## 4. Checkpoint Resuming & Gradient Accumulation
* **Gradient Accumulation:** Emulated batch size $256$ using physical batch size $64$ with accumulation steps $= 4$, stabilizing convergence identically to larger physical batches.
* **Resuming Training:** Interrupted training at epoch 3 and resumed seamlessly from saved checkpoints without loss discontinuity.