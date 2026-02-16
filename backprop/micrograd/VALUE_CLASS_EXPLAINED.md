# The `Value` Class: Automatic Differentiation from Scratch

This document explains every component of the `Value` class in `engineeeee.py` -- a minimal autograd engine that powers backpropagation, the algorithm at the heart of training neural networks.

---

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [What Problem Does This Solve?](#what-problem-does-this-solve)
3. [The Computational Graph](#the-computational-graph)
4. [Constructor: `__init__`](#constructor-__init__)
5. [Forward Pass Operations](#forward-pass-operations)
   - [Addition (`__add__`)](#addition-__add__)
   - [Multiplication (`__mul__`)](#multiplication-__mul__)
   - [Power (`__pow__`)](#power-__pow__)
   - [ReLU (`relu`)](#relu-relu)
6. [The Backward Pass: `backward()`](#the-backward-pass-backward)
7. [Convenience Operators](#convenience-operators)
8. [Full Worked Example](#full-worked-example)
9. [How This Connects to PyTorch](#how-this-connects-to-pytorch)

---

## The Big Picture

Training a neural network requires two things:

1. **Forward pass** -- compute the output (prediction) given inputs and weights.
2. **Backward pass** -- compute how much each weight contributed to the error, so we can update them (gradients).

The `Value` class wraps a single number and automatically tracks **both** of these. Every time you do math with `Value` objects, it silently builds a graph of operations. When you call `.backward()`, it walks that graph in reverse and computes all gradients using the **chain rule** from calculus.

This is exactly what PyTorch's `torch.Tensor` does with `requires_grad=True`, but stripped down to ~90 lines so you can see the mechanism clearly.

---

## What Problem Does This Solve?

Consider a simple function:

```
L = (a * b + c) ** 2
```

If you want to know dL/da, dL/db, dL/dc, you could derive them by hand. But in a neural network with millions of parameters, that is impossible. **Automatic differentiation** solves this by:

- Recording every operation as it happens (building a graph)
- Replaying those operations in reverse, applying the chain rule at each step

The `Value` class implements this pattern.

---

## The Computational Graph

Every `Value` is a **node** in a directed acyclic graph (DAG). Edges point from inputs to outputs.

```
Example: L = (a * b) + c

    a ----\
           (*) --> d ----\
    b ----/               (+) --> L
                  c ----/
```

Each node stores:
- `data` -- the actual numeric value
- `grad` -- the derivative of the final output with respect to this node (dL/d_node)
- `_prev` -- the set of input nodes (parent edges in the graph)
- `_op` -- which operation created this node (for debugging)
- `_backward` -- a function that propagates gradients to `_prev` nodes

---

## Constructor: `__init__`

```python
def __init__(self, data, _children=(), _op=''):
    self.data = data
    self.grad = 0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
```

| Attribute    | Type       | Purpose |
|-------------|------------|---------|
| `data`      | `float`    | The actual scalar value this node holds |
| `grad`      | `float`    | Gradient of the loss with respect to this node. Initialized to 0; filled in during `backward()` |
| `_backward` | `function` | A closure that computes gradient contributions to this node's parents. Default is a no-op (leaf nodes have no parents to propagate to) |
| `_prev`     | `set`      | The set of `Value` nodes that were inputs to the operation that created this node. Empty for leaf nodes (inputs/weights) |
| `_op`       | `str`      | Label for the operation ('+', '*', 'ReLU', etc.). Only used for visualization/debugging |

**Key insight**: Leaf nodes (your raw inputs and weights) start with `_backward = lambda: None` and `_prev = {}` because they have no parent operations. Intermediate nodes created by operations will override `_backward` with the actual gradient logic.

---

## Forward Pass Operations

Each operation does three things:
1. **Compute the result** (forward pass math)
2. **Create a new `Value` node** that records its inputs as children
3. **Define a `_backward` closure** that knows how to propagate gradients through this specific operation

### Addition: `__add__`

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
        self.grad += out.grad
        other.grad += out.grad
    out._backward = _backward

    return out
```

**Forward**: `out.data = self.data + other.data`

**Backward (chain rule)**:

If `out = a + b`, then from calculus:
- d(out)/da = 1
- d(out)/db = 1

By the chain rule: `dL/da = dL/d(out) * d(out)/da = out.grad * 1`

So both parents simply receive `out.grad` as their gradient contribution.

**Why `+=` and not `=`?** A node might be used in multiple operations. For example if `a` appears in both `a + b` and `a * c`, its gradient accumulates contributions from both paths. This is the **multivariate chain rule**: dL/da = dL/da (via path 1) + dL/da (via path 2).

**Why `isinstance` check?** So you can write `a + 2` where `2` is a plain Python number. It gets auto-wrapped into `Value(2)`.

### Multiplication: `__mul__`

```python
def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad
    out._backward = _backward

    return out
```

**Forward**: `out.data = self.data * other.data`

**Backward (chain rule)**:

If `out = a * b`, then:
- d(out)/da = b
- d(out)/db = a

So:
- `a.grad += b.data * out.grad`  (the "local gradient" b times the "upstream gradient" out.grad)
- `b.grad += a.data * out.grad`

This is the core pattern: **local gradient x upstream gradient**.

### Power: `__pow__`

```python
def __pow__(self, other):
    assert isinstance(other, (int, float))
    out = Value(self.data**other, (self,), f'**{other}')

    def _backward():
        self.grad += (other * self.data**(other-1)) * out.grad
    out._backward = _backward

    return out
```

**Forward**: `out.data = self.data ** other`

**Backward**:

If `out = a^n`, then from the power rule:
- d(out)/da = n * a^(n-1)

So: `a.grad += n * a^(n-1) * out.grad`

Note: `other` here is a plain number (the exponent), not a `Value`. Only `self` (the base) has a gradient computed. This is why there is only one child in `(self,)`.

### ReLU: `relu`

```python
def relu(self):
    out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

    def _backward():
        self.grad += (out.data > 0) * out.grad
    out._backward = _backward

    return out
```

**Forward**: `out.data = max(0, self.data)`

This is the Rectified Linear Unit activation function, the most commonly used activation in modern deep learning.

```
ReLU(x) = { 0   if x < 0
           { x   if x >= 0
```

**Backward**:

The derivative of ReLU is a step function:
- If `x > 0`: derivative = 1, so gradient passes through unchanged
- If `x <= 0`: derivative = 0, so gradient is killed (blocked)

The expression `(out.data > 0)` evaluates to `True` (which Python treats as `1`) or `False` (`0`), acting as a gate.

This "gradient gating" is why ReLU can cause **dead neurons** -- if a neuron's output is always negative, its gradient is always 0 and it can never recover during training.

---

## The Backward Pass: `backward()`

This is the most important method. It orchestrates the entire gradient computation.

```python
def backward(self):
    # Step 1: Topological sort
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)

    # Step 2: Backpropagate
    self.grad = 1
    for v in reversed(topo):
        v._backward()
```

### Step 1: Topological Sort

Before we can propagate gradients, we need to process nodes in the right order. If `L = (a * b) + c`, we must compute dL/d(a*b) before we can compute dL/da and dL/db.

**Topological sort** gives us an ordering where every node appears after all its children (inputs). Reversed, this means every node appears **before** its children -- exactly the order we need for backpropagation.

The algorithm is a depth-first search (DFS):
1. For each node, recursively visit all its children first
2. After all children are visited, append the current node
3. Track visited nodes to avoid processing duplicates

```
For L = (a * b) + c:

topo (after sort):   [a, b, (a*b), c, L]
reversed(topo):      [L, c, (a*b), b, a]   <-- backprop order
```

### Step 2: Seed and Propagate

```python
self.grad = 1
```

The gradient of the output with respect to itself is 1 (dL/dL = 1). This is the "seed" that starts the chain rule cascade.

```python
for v in reversed(topo):
    v._backward()
```

Walk the reversed topological order. Each node's `_backward()` function pushes gradient to its parents using the chain rule. By the time we reach a node, all nodes that depend on it have already propagated their gradients to it, so `v.grad` contains the complete accumulated gradient.

---

## Convenience Operators

These methods let you write natural math expressions. They are all implemented by reducing to the core operations above.

```python
def __neg__(self):          # -self
    return self * -1

def __radd__(self, other):  # other + self  (e.g., 2 + a)
    return self + other

def __sub__(self, other):   # self - other
    return self + (-other)

def __rsub__(self, other):  # other - self
    return other + (-self)

def __rmul__(self, other):  # other * self  (e.g., 2 * a)
    return self * other

def __truediv__(self, other):   # self / other
    return self * other**-1

def __rtruediv__(self, other):  # other / self
    return other * self**-1
```

### Why `__radd__` and `__rmul__`?

Python calls `__radd__` when the **left** operand does not support the operation. For `2 + a`:
1. Python tries `int.__add__(2, a)` -- fails (int does not know about `Value`)
2. Python falls back to `Value.__radd__(a, 2)` -- succeeds

Without `__radd__`, expressions like `2 + a` would crash.

### Why implement subtraction and division this way?

Instead of writing new backward functions, they reuse existing ones:
- `a - b` becomes `a + (-b)` which is `a + (b * -1)` -- uses `__mul__` and `__add__`
- `a / b` becomes `a * b^(-1)` -- uses `__pow__` and `__mul__`

This is elegant: only 4 operations (`+`, `*`, `**`, `relu`) need custom backward logic, and all other math is expressed in terms of them.

### `__repr__`

```python
def __repr__(self):
    return f"Value(data={self.data}, grad={self.grad})"
```

Controls what you see when you `print()` a Value. Shows both the current data and its gradient.

---

## Full Worked Example

Let us trace through a complete forward and backward pass.

```python
a = Value(2.0)    # leaf node
b = Value(-3.0)   # leaf node
c = Value(10.0)   # leaf node

d = a * b          # d = 2.0 * -3.0 = -6.0
e = d + c          # e = -6.0 + 10.0 = 4.0
L = e.relu()       # L = max(0, 4.0) = 4.0
```

**Computational graph built during forward pass:**

```
a(2.0) ---\
           (*) --> d(-6.0) ---\
b(-3.0) --/                    (+) --> e(4.0) --> [ReLU] --> L(4.0)
                   c(10.0) --/
```

**Now call `L.backward()`:**

Topological order: `[a, b, d, c, e, L]`
Reversed: `[L, e, c, d, b, a]`

| Step | Node | Action | Result |
|------|------|--------|--------|
| 0 | L | Seed: `L.grad = 1` | L.grad = 1.0 |
| 1 | L | ReLU backward: `e.grad += (4.0 > 0) * 1.0` | e.grad = 1.0 |
| 2 | e | Add backward: `d.grad += 1.0`, `c.grad += 1.0` | d.grad = 1.0, c.grad = 1.0 |
| 3 | c | Leaf node, `_backward` is no-op | -- |
| 4 | d | Mul backward: `a.grad += (-3.0) * 1.0`, `b.grad += (2.0) * 1.0` | a.grad = -3.0, b.grad = 2.0 |
| 5 | b | Leaf node, no-op | -- |
| 6 | a | Leaf node, no-op | -- |

**Final gradients:**

| Node | data | grad | Meaning |
|------|------|------|---------|
| a | 2.0 | -3.0 | If a increases by h, L decreases by 3h |
| b | -3.0 | 2.0 | If b increases by h, L increases by 2h |
| c | 10.0 | 1.0 | If c increases by h, L increases by h |

You can verify by hand: L = relu(a*b + c), so dL/da = b = -3.0 (when relu is active).

---

## How This Connects to PyTorch

The exact same computation in PyTorch:

```python
import torch

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(-3.0, requires_grad=True)
c = torch.tensor(10.0, requires_grad=True)

L = (a * b + c).relu()
L.backward()

print(a.grad)  # -3.0
print(b.grad)  #  2.0
print(c.grad)  #  1.0
```

PyTorch's autograd engine does the same thing as this `Value` class, but:
- Operates on **tensors** (multi-dimensional arrays) instead of scalars
- Has hundreds of operations with custom backward functions
- Uses C++/CUDA for GPU acceleration
- Supports higher-order gradients, checkpointing, and many optimizations

But the fundamental idea is identical: build a graph during the forward pass, walk it in reverse during the backward pass, apply the chain rule at each node.

---

## Summary: The Chain Rule in Code

The entire class boils down to one idea from calculus:

```
dL/dx = dL/dy * dy/dx
         ^        ^
    "upstream"  "local"
     gradient   gradient
```

Every `_backward()` function computes `local_gradient * upstream_gradient` and adds it to the parent's `.grad`. The `backward()` method ensures nodes are processed in the right order so that upstream gradients are always ready when needed.

That is all backpropagation is. Everything else in deep learning frameworks is engineering built on top of this core idea.
