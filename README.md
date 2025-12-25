# torchnd

N-dimensional convolution for PyTorch that works with any number of spatial dimensions, supporting groups, dilation, transposed convolution, complex numbers, and arbitrary dimension layouts.

## Installation

```bash
pip install torchnd
```

## Usage

**2D convolution:**
```python
import torch
from torchnd import conv_nd

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

**Modules:**
```python
from torchnd import ConvNd, ConvTransposeNd

conv = ConvNd(4, 8, 3, dim=(-2, -1), padding=1)
out = conv(x)
```

**Padding:**
```python
from torchnd import pad_nd, adjoint_pad_nd

# Pad or crop arbitrary dimensions
padded = pad_nd(x, pad=(1, 1, 2, 2), dims=(-2, -1), mode="reflect")
cropped = pad_nd(x, pad=(-1, -1), dims=(0,))  # Negative values crop

# Adjoint of padding (unpads)
unpadded = adjoint_pad_nd(padded, pad=(1, 1, 2, 2), dims=(-2, -1))
```

**Adjoint operators:**
```python
from torchnd import ConvNd

conv = ConvNd(4, 8, 3, dim=(-2, -1), padding=1, bias=False)
x = torch.randn(2, 4, 16, 16)
y = torch.randn_like(conv(x))

# Adjoint satisfies: <conv(x), y> = <x, conv.adjoint(y)>
# For ConvNd: adjoint is transposed convolution with conjugated weights
adj_y = conv.adjoint(y, input_shape=(16, 16))
```

## Functions

### pad_nd

N-dimensional padding and cropping with flexible dimension specification. Supports arbitrary dimension layouts, not limited to the last N dimensions. Negative padding values crop the tensor.

Modes: `constant`, `reflect`, `replicate`, `circular`. Each dimension can use a different mode. For constant mode, negative padding crops symmetrically. For non-constant modes, negative padding is not supported.

The `dims` parameter specifies which dimensions to pad. If `None`, pads the last N dimensions where N = len(pad) // 2. Padding is specified as pairs (left, right) per dimension.

### adjoint_pad_nd

Computes the adjoint of `pad_nd` via autograd. For a padding operator P, the adjoint P* satisfies the inner product identity: `<Px, y> = <x, P*y>` for all x, y. The adjoint unpads the tensor, summing contributions from padded regions back into the original shape. For constant/zeros mode, the adjoint is equivalent to cropping. For other modes, it correctly handles the adjoint of the boundary extension.

### Adjoint convolution

The `ConvNd` and `ConvTransposeNd` modules provide `adjoint()` methods. For `ConvNd`, the adjoint is the transposed convolution with conjugated weights (for complex) or identical weights (for real). For `ConvTransposeNd`, the adjoint is the forward convolution with conjugated weights.

The adjoint satisfies the inner product identity: `<Ax, y> = <x, A*y>` where A is the convolution operator and A* is its adjoint. This is verified via dot product tests. The adjoint is undefined when bias is present.

## Implementation

For dimensions beyond 3D, `conv_nd` recursively decomposes the convolution into lower-dimensional operations. A 4D convolution becomes a sum of 3D convolutions applied to strided slices of the input.

Complex convolution is handled by decomposing into real operations: `(a+bi)*(c+di) = (ac-bd) + (ad+bc)i`.

The implementation uses native PyTorch operations when possible (1D, 2D, 3D) and falls back to recursion only for higher dimensions, minimizing memory overhead with strided views.
