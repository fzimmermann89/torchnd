## Project Stack

- Python 3.12
- torch 2.6+
- Linting: ruff, mypy
- Testing: pytest
- Package manager: uv

## Environment

**Always activate venv before running any tool or script:**

```bash
source .venv/bin/activate
```

```bash
# Linting/type checking
ruff check .
ruff format .
mypy .

# Tests
pytest
pytest tests/test_models/test_unet.py -v

## Code Style

### General

- Pythonic, concise. No boilerplate.
- `pathlib.Path` over `os.path`.
- f-strings for formatting.
- Modern syntax: `list[T]`, `dict[K, V]`, `X | None`.

### Type Hints

- All function signatures typed.
- `torch.Tensor` for tensors.
- Annotate shapes in comments: `# (B, C, H, W)`.

```python
def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # (B,C,H,W), (B,) -> (B,C,H,W)
    ...
```

### Docstrings

- NumPy style. NO types in docstrings.
- `__init__` parameters documented in `__init__` docstring.
- Document shapes, expected ranges, semantics.

```python
class UNet(torch.nn.Module):
    """UNet with FiLM conditioning for diffusion."""

    def __init__(self, in_channels: int, base_dim: int = 64, time_emb_dim: int = 256):
        """
        Parameters
        ----------
        in_channels
            Input channels (2 for complex: real/imag).
        base_dim
            Base feature dimension, doubled at each downsampling.
        time_emb_dim
            Dimension of sinusoidal time embedding.
        """
        super().__init__()
        ...
```

### Comments

- NO trivial comments.
- Only explain non-obvious logic, numerical tricks, paper references.

```python
# Apply noise schedule from Ho et al. (2020) eq. 4
alpha_bar = torch.cumprod(1 - betas, dim=0)
```

## PyTorch Guidelines

### Performance

- Vectorize—no Python loops over batch/spatial dims.
- `torch.compile()` for hot paths.
- In-place ops when safe: `x.add_(1)`
- `pin_memory=True`, `num_workers>0` in DataLoader.
- Mixed precision via Lightning `precision="16-mixed"`.

### einops

```python
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

# Good
x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=16, p2=16)
t_emb = repeat(t_emb, "b d -> b d h w", h=h, w=w)

# Avoid
x = x.view(...).permute(...).reshape(...)
```

make use of unflatten instead of view.
prefer moveaxis instead of permute (and einops.rearrange over permute)


## Testing

- Mirror main structure in `tests/`.
- Use fixtures in `conftest.py` for common tensors/models.
- Test shapes, dtypes, gradient flow, edge cases.


## Imports

- Do not import `torch.nn as nn` or `torch.nn.functional as F`. Use full path in the code.
