import numpy as np

class Value:
    """Stores a scalar value and its associated gradient for autograd."""
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __neg__(self):
        return self * -1.0

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data ** other, (self,), f'**{other}')
        
        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), 'ReLU')
        
        def _backward():
            self.grad += (1.0 if self.data > 0.0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        t = 1.0 / (1.0 + np.exp(-self.data))
        out = Value(t, (self,), 'Sigmoid')
        
        def _backward():
            self.grad += t * (1.0 - t) * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(np.log(max(self.data, 1e-15)), (self,), 'Log')
        
        def _backward():
            self.grad += (1.0 / max(self.data, 1e-15)) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # Topological sort of all nodes in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Initialize output gradient to 1.0 and backpropagate
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()