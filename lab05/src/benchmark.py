import torch
import numpy as np
from engine import Value

def verify_gradients():
    a_val, b_val, c_val = 2.0, 1.5, -1.0

    # 1. PyTorch Autograd
    a_t = torch.tensor(a_val, requires_grad=True, dtype=torch.float64)
    b_t = torch.tensor(b_val, requires_grad=True, dtype=torch.float64)
    c_t = torch.tensor(c_val, requires_grad=True, dtype=torch.float64)
    res_t = ((a_t * b_t + c_t) ** 2).relu().sum()
    res_t.backward()

    # 2. Custom Autograd Engine
    a_c = Value(a_val)
    b_c = Value(b_val)
    c_c = Value(c_val)
    res_c = ((a_c * b_c + c_c) ** 2).relu()
    res_c.backward()

    # 3. Numerical Gradients (Finite Differences)
    eps = 1e-5
    def f(a, b, c):
        return max(0.0, (a * b + c) ** 2)
    
    num_a = (f(a_val + eps, b_val, c_val) - f(a_val - eps, b_val, c_val)) / (2 * eps)
    num_b = (f(a_val, b_val + eps, c_val) - f(a_val, b_val - eps, c_val)) / (2 * eps)
    num_c = (f(a_val, b_val, c_val + eps) - f(a_val, b_val, c_val - eps)) / (2 * eps)

    diff = max(abs(a_c.grad - a_t.grad.item()),
               abs(b_c.grad - b_t.grad.item()),
               abs(c_c.grad - c_t.grad.item()))

    print(f"PyTorch Gradients:    a={a_t.grad.item():.4f}, b={b_t.grad.item():.4f}, c={c_t.grad.item():.4f}")
    print(f"Custom Gradients:     a={a_c.grad:.4f}, b={b_c.grad:.4f}, c={c_c.grad:.4f}")
    print(f"Numerical Gradients:  a={num_a:.4f}, b={num_b:.4f}, c={num_c:.4f}")
    print(f"Max Absolute Difference: {diff:.2e}")
    assert diff < 1e-6, "Gradient mismatch exceeds 1e-6 threshold!"
    print("Verification successfully passed (< 1e-6)!")

if __name__ == "__main__":
    verify_gradients()