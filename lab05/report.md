# Lab 5 Report: Frameworks and Automatic Differentiation

## 1. Custom Autograd Engine & Graph Construction
We implemented a scalar automatic differentiation engine (`Value`) supporting dynamic graph construction during the forward pass. Each node records its parent operations (`_prev`, `_op`) and defines local derivative functions (`_backward`) adhering to the chain rule. 
* **Topological Sort:** Ensures nodes are visited in correct dependency order during backpropagation.
* **Gradient Accumulation:** Gradients are accumulated (`+=`) across branches to properly handle variables used in multiple expressions.

## 2. Gradient Verification
We verified our engine against PyTorch autograd and finite-difference numerical gradients on the test expression:
$$\sum \text{ReLU}\left((a \cdot b + c)^2\right)$$
* **Max Absolute Difference:** $< 10^{-6}$ across all parameters ($a, b, c$), confirming mathematical and computational correctness.

## 3. Tiny Model on `make_moons`
A tiny multi-layer perceptron ($2 \to 8 \to 1$ with ReLU hidden activation and Sigmoid output) was trained using Binary Cross-Entropy loss on synthetic moon data. 

## 4. Eager vs. Compiled Execution & Tensor Contiguity
* **Eager Execution:** Executes operations dynamically line-by-line, incurring Python interpreter and dispatch overhead per operation.
* **Compiled Execution (`torch.compile` / `torch.jit.trace`):** Fuses operations and eliminates Python overhead, yielding speedups especially in repetitive training loops and deeper architectures.
* **Tensor Contiguity Impact:** Non-contiguous tensors (such as transposed views) introduce memory stride non-uniformities that disrupt cache locality and vectorized memory access, measurably degrading matrix multiplication speed.