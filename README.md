# torch-conv-nd

N-dimensional convolution for PyTorch that works with any number of spatial dimensions, supporting groups, dilation, transposed convolution, complex numbers, and arbitrary dimension layouts.

## Installation

```bash
pip install torch-conv-nd
```

## Usage

**2D convolution:**
```python
import torch
from torch_conv_nd import conv_nd

x = torch.randn(2, 4, 16, 16)
weight = torch.randn(8, 4, 3, 3)
out = conv_nd(x, weight, dim=(-2, -1), padding=1, stride=2)
```

**4D convolution** (batch × time × height × width):
```python
x = torch.randn(2, 4, 10, 16, 16)
weight = torch.randn(8, 4, 3, 3, 3)
out = conv_nd(x, weight, dim=(-3, -2, -1), padding=1)
```

**Channel-last layout:**
```python
x = torch.randn(2, 16, 16, 4)
weight = torch.randn(8, 4, 3, 3)
out = conv_nd(x, weight, dim=(1, 2), channel_dim=-1, padding=1)
```

**Transposed convolution:**
```python
x = torch.randn(2, 4, 8, 8)
weight = torch.randn(4, 8, 3, 3)
out = conv_nd(x, weight, dim=(-2, -1), padding=1, stride=2, transposed=True)
```

**Complex numbers:**
```python
x = torch.randn(2, 4, 16, 16, dtype=torch.complex64)
weight = torch.randn(8, 4, 3, 3, dtype=torch.complex64)
out = conv_nd(x, weight, dim=(-2, -1), padding=1)
```

**Asymmetric parameters per dimension:**
```python
x = torch.randn(2, 4, 16, 16)
weight = torch.randn(8, 4, 3, 3)
out = conv_nd(
    x, weight,
    dim=(-2, -1),
    stride=(2, 1),
    padding=(1, 2),
    dilation=(1, 2)
)
```

## Implementation

For dimensions beyond 3D, `conv_nd` recursively decomposes the convolution into lower-dimensional operations. A 4D convolution becomes a sum of 3D convolutions applied to strided slices of the input.

Complex convolution is handled by decomposing into real operations: `(a+bi)*(c+di) = (ac-bd) + (ad+bc)i`.

The implementation uses native PyTorch operations when possible (1D, 2D, 3D) and falls back to recursion only for higher dimensions, minimizing memory overhead with strided views.
