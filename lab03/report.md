# Lab 3: Architectures and Their Costs

## 1. Goal
Dataset & Split: Fashion-MNIST loaded via torchvision.datasets, normalized using mean (0.2860,) and std (0.3530,). Official 60k training set split into 55k train and 5k validation; official 10k set used for testing. Batch size set to 128.

## 2. Method
Architectures:MLP: 
Flatten(784) -> Linear(784, 256) -> ReLU -> Linear(256, 128) -> ReLU -> Linear(128, 10).CNN: Conv2d(1, 16, k=3, p=1) -> ReLU -> MaxPool2d(2) -> Conv2d(16, 32, k=3, p=1) -> ReLU -> MaxPool2d(2) -> Flatten -> Linear(1568, 10).
Attention: $28 \times 28$ image split into 16 patches of $7 \times 7$ via Conv2d(1, 64, k=7, s=7). Positional embeddings added, processed through a 2-layer Transformer encoder (d_model=64, nhead=4, dim_feedforward=128), mean pooled, and classified via Linear(64, 10).

Training: Trained for 5 epochs using Adam optimizer (lr=1e-3) with Cross-Entropy loss.

## 3. Results Table

Model 235,146	|  87.85%	|  470,292	| 0.082	 |  0.312

MLP4  20,810	|  89.44%	|  3,114,240	| 0.814 |  7.848

CNN   71,946	|  85.53%	|  3,212,544	| 0.925 |  7.872

Attention
## 4. Conclusions
Most Cost-Effective Architecture: The CNN emerges as the most cost-effective architecture overall. Despite using only ~20.8k parameters (less than 10% of the MLP parameter budget), it achieved the highest test accuracy (89.44%).

Impact of Spatial Locality & Memory Access Patterns:
CNN: Benefited strongly from weight sharing and high spatial data reuse. Convolution kernels are reused across all spatial locations, allowing values to remain in fast on-chip SRAM/cache rather than constantly fetching new parameters from off-chip DRAM.

MLP: Showed low computational efficiency per parameter. Every weight in dense layers is read once per forward pass without spatial reuse, consuming significant memory bandwidth relative to useful calculations.

Attention Model: Self-attention incurs quadratic scaling ($O(N^2)$) relative to token sequence length for matrix multiplications ($QK^T$), introducing intermediate memory bandwidth overhead and higher latency per patch compared to localized convolution kernels.

FlOP Profiling Gaps: Minor gaps between hand calculations and thop arise because manual estimations often ignore bias vector additions, activation layers (ReLU), and normalization passes, while profiling tools vary in whether they count Multiply-Accumulate operations (MACs) as 1 MAC or 2 FLOPs.