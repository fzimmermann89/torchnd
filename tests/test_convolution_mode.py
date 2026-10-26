"""Independent values and adjoints for both kernel conventions."""

from itertools import product

import pytest
import torch

from torchnd import ConvNd, ConvTransposeNd, conv_nd


def reference_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    convolution: bool,
    transposed: bool,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    output_padding: int = 0,
) -> torch.Tensor:
    """Evaluate the defining sums directly, without any convolution backend."""
    spatial = x.shape[2:]
    kernel = weight.shape[2:]
    if transposed:
        out_shape = tuple(
            (n - 1) * stride - 2 * padding + dilation * (k - 1) + 1 + output_padding for n, k in zip(spatial, kernel, strict=True)
        )
        out_channels = weight.shape[1] * groups
    else:
        out_shape = tuple((n + 2 * padding - dilation * (k - 1) - 1) // stride + 1 for n, k in zip(spatial, kernel, strict=True))
        out_channels = weight.shape[0]
    out = x.new_zeros(x.shape[0], out_channels, *out_shape)
    in_per_group = x.shape[1] // groups
    out_per_group = out_channels // groups
    positions = spatial if transposed else out_shape
    for batch, group, ci, co in product(range(x.shape[0]), range(groups), range(in_per_group), range(out_per_group)):
        ic, oc = group * in_per_group + ci, group * out_per_group + co
        for pos, offset in product(product(*(range(n) for n in positions)), product(*(range(k) for k in kernel))):
            target = tuple(n * stride - padding + k * dilation for n, k in zip(pos, offset, strict=True))
            limits = out_shape if transposed else spatial
            if any(i < 0 or i >= n for i, n in zip(target, limits, strict=True)):
                continue
            wi = tuple(k - 1 - i for k, i in zip(kernel, offset, strict=True)) if convolution else offset
            w = weight[(ic, co, *wi)] if transposed else weight[(oc, ci, *wi)]
            if transposed:
                out[(batch, oc, *target)] += x[(batch, ic, *pos)] * w
            else:
                out[(batch, oc, *pos)] += x[(batch, ic, *target)] * w
    return out


@pytest.mark.parametrize("convolution", [False, True])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("ndim", [0, 1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_functional_reference(convolution: bool, transposed: bool, ndim: int, dtype: torch.dtype) -> None:
    """Check grouped, strided, dilated kernels and channel-last layouts."""
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(1, 4, *((3,) * ndim), dtype=dtype, generator=generator)
    weight = torch.randn(4, 2, *((2,) * ndim), dtype=dtype, generator=generator)
    expected = reference_conv(
        x,
        weight,
        convolution=convolution,
        transposed=transposed,
        stride=2,
        padding=1,
        dilation=2,
        groups=2,
        output_padding=int(transposed),
    )
    actual = conv_nd(
        x.moveaxis(1, -1),
        weight,
        dim=tuple(range(1, ndim + 1)),
        channel_dim=-1,
        stride=2,
        padding=1,
        dilation=2,
        groups=2,
        transposed=transposed,
        output_padding=int(transposed),
        convolution=convolution,
    )
    torch.testing.assert_close(actual.moveaxis(-1, 1), expected)
    if not convolution:
        default = conv_nd(
            x.moveaxis(1, -1),
            weight,
            dim=tuple(range(1, ndim + 1)),
            channel_dim=-1,
            stride=2,
            padding=1,
            dilation=2,
            groups=2,
            transposed=transposed,
            output_padding=int(transposed),
        )
        assert torch.equal(actual, default)


@pytest.mark.parametrize("convolution", [False, True])
@pytest.mark.parametrize("conv_type", [ConvNd, ConvTransposeNd])
@pytest.mark.parametrize("ndim", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_module_reference_and_adjoint(
    convolution: bool,
    conv_type: type[ConvNd] | type[ConvTransposeNd],
    ndim: int,
    dtype: torch.dtype,
) -> None:
    """Modules and their adjoints agree with the independent matrix entries."""
    generator = torch.Generator().manual_seed(5)
    x = torch.randn(1, 4, *((3,) * ndim), dtype=dtype, generator=generator, requires_grad=True)
    weight = torch.randn(4, 2, *((2,) * ndim), dtype=dtype, generator=generator)
    module = conv_type(
        4,
        4,
        2,
        dim=tuple(range(2, ndim + 2)),
        stride=2,
        padding=1,
        groups=2,
        bias=False,
        weight=weight,
        convolution=convolution,
    )
    out = module(x)
    expected = reference_conv(x, weight, convolution=convolution, transposed=module.transposed, stride=2, padding=1, groups=2)
    torch.testing.assert_close(out, expected)
    y = torch.randn(out.shape, dtype=dtype, generator=generator)
    adjoint = module.adjoint(y, input_shape=(3,) * ndim) if isinstance(module, ConvNd) else module.adjoint(y)
    (expected_adjoint,) = torch.autograd.grad(expected, x, y)
    torch.testing.assert_close(adjoint, expected_adjoint)
    torch.testing.assert_close(torch.vdot(out.flatten(), y.flatten()), torch.vdot(x.flatten(), adjoint.flatten()))


@pytest.mark.parametrize("convolution", [False, True])
@pytest.mark.parametrize("conv_type", [ConvNd, ConvTransposeNd])
@pytest.mark.parametrize("padding_mode", ["zeros", "reflect", "replicate", "circular"])
def test_padded_adjoint(
    convolution: bool,
    conv_type: type[ConvNd] | type[ConvTransposeNd],
    padding_mode: str,
) -> None:
    """Boundary extension and its adjoint retain the chosen kernel convention."""
    generator = torch.Generator().manual_seed(8)
    module = conv_type(
        2, 2, 3, dim=(-1,), padding=1, padding_mode=padding_mode, bias=False, dtype=torch.complex128, convolution=convolution
    )
    x = torch.randn(1, 2, 5, dtype=torch.complex128, generator=generator, requires_grad=True)
    out = module(x)
    y = torch.randn(out.shape, dtype=out.dtype, generator=generator)
    (expected,) = torch.autograd.grad(out, x, y)
    torch.testing.assert_close(module.adjoint(y), expected)


@pytest.mark.parametrize("conv_type", [ConvNd, ConvTransposeNd])
@pytest.mark.parametrize("external", ["internal", "parameter", "buffer"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))]
)
def test_weight_updates(conv_type: type[ConvNd] | type[ConvTransposeNd], external: str, dtype: torch.dtype, device: str) -> None:
    """Each call uses the current weight and builds a fresh differentiable flip."""
    weight = torch.arange(1.0, 4.0, device=device).to(dtype).unflatten(0, (1, 1, 3))
    if dtype.is_complex:
        weight = weight + 0.5j * weight.square()
    if external == "parameter":
        weight = torch.nn.Parameter(weight)
    else:
        weight.requires_grad_()
    module = conv_type(
        1,
        1,
        3,
        dim=(-1,),
        bias=False,
        device=torch.device(device),
        dtype=dtype,
        weight=None if external == "internal" else weight,
        convolution=True,
    )
    if external == "internal":
        weight = module.weight
    assert module.weight is weight
    if external != "internal":
        assert dict(module.named_buffers())["weight"] is weight
    else:
        assert dict(module.named_parameters())["weight"] is weight
    x = torch.tensor([[[1.0, 2.0, 4.0, 8.0]]], dtype=dtype, device=device)
    for step in range(2):
        out = module(x)
        reference_weight = weight.detach().clone().requires_grad_()
        expected = reference_conv(x, reference_weight, convolution=True, transposed=module.transposed)
        torch.testing.assert_close(out, expected)
        out.abs().square().sum().backward()
        (expected_grad,) = torch.autograd.grad(expected.abs().square().sum(), reference_weight)
        torch.testing.assert_close(weight.grad, expected_grad)
        y = torch.ones_like(out)
        weight.grad = None
        module.adjoint(y).abs().square().sum().backward()
        expected_adjoint = reference_conv(y, reference_weight.conj(), convolution=True, transposed=not module.transposed)
        (expected_grad,) = torch.autograd.grad(expected_adjoint.abs().square().sum(), reference_weight)
        torch.testing.assert_close(weight.grad, expected_grad)
        assert weight.grad is not None
        assert torch.isfinite(weight.grad).all()
        torch.testing.assert_close(module.state_dict()["weight"], weight)
        with torch.no_grad():
            weight.add_(step + 1)
        weight.grad = None
    assert "convolution=True" in repr(module)
