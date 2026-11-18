# Individual Project Proposal: Efficient Edge Image Classification

* **Student Name:** Volkov Illia
* **Student ID:** 76176
* **Problem:** Deploying deep convolutional neural networks on resource-constrained edge devices (such as Raspberry Pi or mobile hardware) while maintaining real-time classification performance.
* **Dataset:** CIFAR-10 (32x32 color images across 10 classes).
* **Success Metric:** Top-1 validation accuracy $\geq 85\%$ with inference latency $\leq 50\text{ ms}$ and model size $\leq 5\text{ MB}$[cite: 14].
* **Deployment Constraints:** Edge memory footprint $\leq 5\text{ MB}$, single-thread CPU inference latency under $50\text{ ms}$[cite: 14].
* **Planned Optimization:** Post-training quantization (INT8) and weight pruning combined with structural knowledge distillation to compress model size[cite: 14].