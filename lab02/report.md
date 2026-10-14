# Lab 2: A Neural Network from Scratch

## 1. Goal
To design and implement a multi-layer perceptron (two hidden layers) entirely from scratch using NumPy, validate gradient updates via numerical gradient checking, evaluate hyperparameter configurations (learning rates and hidden widths), and profile execution timings.

## 2. Method
* **Dataset**: Optical recognition of handwritten digits (`load_digits`), inputs scaled by $16.0$, labels one-hot encoded, and split 70/30 with stratification (`random_state=42`).
* **Architecture**: Input ($64$) $\rightarrow$ Hidden Layer 1 ($W$) $\rightarrow$ ReLU $\rightarrow$ Hidden Layer 2 ($W$) $\rightarrow$ ReLU $\rightarrow$ Output ($10$) $\rightarrow$ Softmax.
* **Initialization**: Weights drawn from $N(0, 0.01)$ and biases set to zeros.
* **Optimization**: Mini-batch stochastic gradient descent (SGD) with batch size $32$ over $50$ epochs.

## 3. Results & Hyperparameter Sweeps

| Hidden Width ($W$) | Learning Rate ($\eta$) | Final Test Accuracy |
| :--- | :--- | :--- |
| 16 | 0.001 | [Insert from terminal] |
| 16 | 0.01 | [Insert from terminal] |
| 16 | 0.1 | [Insert from terminal] |
| 64 | 0.001 | [Insert from terminal] |
| 64 | 0.01 | [Insert from terminal] |
| 64 | 0.1 | [Insert from terminal] |

*Loss curves have been successfully generated and saved to `lab02/results/loss_curves.png`.*

## 4. Execution Timing Breakdown
* **Median Forward Pass Latency**: [Insert from terminal] ms
* **Median Backward Pass Latency**: [Insert from terminal] ms

**Analysis**: The backward pass typically takes longer than the forward pass because it requires computing gradients with respect to activations, weights, and biases across all layers via the chain rule, involving multiple matrix transposition and dot-product operations.