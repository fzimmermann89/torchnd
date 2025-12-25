"""N-dimensional convolution with flexible dimension specification."""

from collections.abc import Callable, Sequence

import torch
from einops import rearrange
from torch import Tensor

CONV_REGISTRY: dict[int, Callable[..., Tensor]] = {
    1: torch.nn.functional.conv1d,
    2: torch.nn.functional.conv2d,
    3: torch.nn.functional.conv3d,
}

CONV_TRANSPOSE_REGISTRY: dict[int, Callable[..., Tensor]] = {
    1: torch.nn.functional.conv_transpose1d,
    2: torch.nn.functional.conv_transpose2d,
    3: torch.nn.functional.conv_transpose3d,
}


def _normalize_tuple(value: int | tuple[int, ...], length: int) -> tuple[int, ...]:
    return (value,) * length if isinstance(value, int) else tuple(value)


def conv_nd(
    x: Tensor,
    weight: Tensor,
    dim: tuple[int, ...] = (-2, -1),
    channel_dim: int = 1,
    stride: int | tuple[int, ...] = 1,
    padding: int | tuple[int, ...] = 0,
    dilation: int | tuple[int, ...] = 1,
    groups: int = 1,
    transposed: bool = False,
    output_padding: int | tuple[int, ...] = 0,
) -> Tensor:
    """N-dimensional convolution with flexible dimension specification.

    Supports arbitrary spatial dimensions via recursive decomposition when
    native PyTorch implementations (1D, 2D, 3D) are unavailable.

    Note: Strictly speaking, this implements correlation, not convolution. But it matches pytorch's naming convention.

    Parameters
    ----------
    x
        Input tensor with batch, channel, and spatial dimensions.
    weight
        Convolution kernel.
        Forward: ``(C_out, C_in // groups, *kernel_sizes)``.
        Transposed: ``(C_in, C_out // groups, *kernel_sizes)``.
    dim
        Spatial dimensions to convolve over.
    channel_dim
        Channel dimension index.
    stride
        Stride of the convolution; controls the step size the kernel moves along each spatial dimension. Can be a single integer or a tuple giving a value per dimension.
    padding
        Padding added to each spatial dimension. Can be a single integer or a tuple specifying padding for each dimension.
    dilation
        Spacing between kernel elements along each spatial dimension. Can be a single integer or per-dimension tuple.
    groups
        Grouped convolution groups.
    transposed
        If True, perform transposed (fractionally-strided) convolution.
    output_padding
        Additional size for transposed conv output (disambiguates shape).

    Returns
    -------
        Convolved output preserving input's dimension layout.
    """
    num_spatial = len(dim)
    stride = _normalize_tuple(stride, num_spatial)
    padding = _normalize_tuple(padding, num_spatial)
    dilation = _normalize_tuple(dilation, num_spatial)
    output_padding = _normalize_tuple(output_padding, num_spatial)

    def conv(a: Tensor, b: Tensor) -> Tensor:
        return _dispatch(a, b, dim, channel_dim, stride, padding, dilation, groups, transposed, output_padding)

    match (x.is_complex(), weight.is_complex()):
        case (True, True):
            return torch.complex(
                conv(x.real, weight.real) - conv(x.imag, weight.imag),
                conv(x.real, weight.imag) + conv(x.imag, weight.real),
            )
        case (True, False):
            return torch.complex(conv(x.real, weight), conv(x.imag, weight))
        case (False, True):
            return torch.complex(conv(x, weight.real), conv(x, weight.imag))
        case _:
            return conv(x, weight)


def _dispatch(
    x: Tensor,
    weight: Tensor,
    dim: tuple[int, ...],
    channel_dim: int,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
    transposed: bool,
    output_padding: tuple[int, ...],
) -> Tensor:
    """Permute to standard layout, convolve, permute back."""
    ndim = x.ndim
    spatial_dims = tuple(d % ndim for d in dim)
    channel_dim_normalized = channel_dim % ndim
    batch_dims = tuple(d for d in range(ndim) if d not in set(spatial_dims) and d != channel_dim_normalized)

    to_standard = batch_dims + (channel_dim_normalized,) + spatial_dims
    x_perm = x.permute(to_standard)

    num_batch = len(batch_dims)
    batch_shape = x_perm.shape[:num_batch]
    x_flat = x_perm.reshape(-1 if batch_shape else 1, *x_perm.shape[num_batch:])

    out_flat = _conv_core(x_flat, weight, stride, padding, dilation, groups, transposed, output_padding)

    out = out_flat.reshape(*batch_shape, *out_flat.shape[1:]) if batch_shape else out_flat.squeeze(0)
    from_standard = torch.argsort(torch.tensor(to_standard)).tolist()
    return out.permute(from_standard)


def _conv_core(
    x: Tensor,
    weight: Tensor,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
    transposed: bool,
    output_padding: tuple[int, ...],
) -> Tensor:
    """Dispatch to native implementation or recursive fallback."""
    num_spatial = len(stride)
    registry = CONV_TRANSPOSE_REGISTRY if transposed else CONV_REGISTRY

    if num_spatial == 0:
        return registry[1](x.unsqueeze(-1), weight.unsqueeze(-1), groups=groups).squeeze(-1)

    if num_spatial in registry:
        kwargs = {
            "bias": None,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
        }
        if transposed:
            kwargs["output_padding"] = output_padding
        return registry[num_spatial](x, weight, **kwargs)

    if transposed:
        return _conv_transpose_recursive(x, weight, stride, padding, dilation, groups, output_padding)
    return _conv_recursive(x, weight, stride, padding, dilation, groups)


def _conv_recursive(
    x: Tensor,
    weight: Tensor,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
) -> Tensor:
    """Decompose N-D conv: sum over first kernel dim of (N-1)-D convs."""
    if padding[0] > 0:
        pad_spec = (0, 0) * (x.ndim - 3) + (padding[0], padding[0])
        x = torch.nn.functional.pad(x, pad_spec)

    kernel_size = weight.shape[2]
    input_size = x.shape[2]
    batch_size = x.shape[0]
    output_size = (input_size - dilation[0] * (kernel_size - 1) - 1) // stride[0] + 1

    accumulator: Tensor | None = None
    for ki in range(kernel_size):
        start = ki * dilation[0]
        x_slice = x[:, :, start : start + (output_size - 1) * stride[0] + 1 : stride[0]]

        out = _conv_core(
            rearrange(x_slice, "n c d ... -> (n d) c ..."),
            weight[:, :, ki],
            stride[1:],
            padding[1:],
            dilation[1:],
            groups,
            False,
            (),
        )
        out = rearrange(out, "(n d) c ... -> n c d ...", n=batch_size, d=output_size)
        accumulator = out if accumulator is None else accumulator + out

    assert accumulator is not None
    return accumulator


def _conv_transpose_recursive(
    x: Tensor,
    weight: Tensor,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
    output_padding: tuple[int, ...],
) -> Tensor:
    """Decompose N-D transpose conv: scatter (N-1)-D results into output."""
    kernel_size = weight.shape[2]
    input_size = x.shape[2]
    batch_size = x.shape[0]
    out_channels = weight.shape[1] * groups
    full_size = (input_size - 1) * stride[0] + dilation[0] * (kernel_size - 1) + 1

    result: Tensor | None = None
    for ki in range(kernel_size):
        out = _conv_core(
            rearrange(x, "n c d ... -> (n d) c ..."),
            weight[:, :, ki],
            stride[1:],
            padding[1:],
            dilation[1:],
            groups,
            True,
            output_padding[1:],
        )
        out = rearrange(out, "(n d) c ... -> n c d ...", n=batch_size, d=input_size)

        if result is None:
            result = x.new_zeros(batch_size, out_channels, full_size, *out.shape[3:])

        start = ki * dilation[0]
        end = start + input_size * stride[0]
        result[:, :, start : end : stride[0]] = result[:, :, start : end : stride[0]] + out

    assert result is not None

    if padding[0] > 0:
        result = result[:, :, padding[0] :]

    right_adjust = padding[0] - output_padding[0]
    if right_adjust > 0:
        result = result[:, :, :-right_adjust]
    elif right_adjust < 0:
        pad_spec = (0, 0) * (result.ndim - 3) + (0, -right_adjust)
        result = torch.nn.functional.pad(result, pad_spec)

    return result


def pad_nd(
    input: Tensor,
    pad: Sequence[int],
    mode: str = "constant",
    value: float = 0.0,
    dims: Sequence[int] | None = None,
) -> Tensor:
    """
    N-dimensional padding/cropping supporting arbitrary dimensions.

    Parameters
    ----------
    input
        Input tensor.
    pad
        Padding sizes as pairs (left, right) per dimension.
        Negative values crop. Without `dims`: PyTorch convention.
    mode
        'constant', 'circular', 'reflect', or 'replicate'.
    value
        Fill value for 'constant' mode.
    dims
        Dimensions to pad. If None, pads last N dimensions.

    Returns
    -------
        Padded/cropped tensor.
    """
    if len(pad) % 2 != 0:
        raise ValueError("pad must have even length")

    n_dims = len(pad) // 2
    if n_dims == 0:
        return input

    ndim = input.ndim

    if dims is None:
        target_dims = list(range(ndim - 1, ndim - n_dims - 1, -1))
        pad_pairs = [(pad[2 * i], pad[2 * i + 1]) for i in range(n_dims)]
    else:
        if len(dims) != n_dims:
            raise ValueError("len(dims) must equal len(pad) // 2")
        target_dims = [d % ndim for d in dims]
        pad_pairs = [(pad[2 * i], pad[2 * i + 1]) for i in range(n_dims)]
        if len(set(target_dims)) != len(target_dims):
            raise ValueError("duplicate dimensions not supported")

    if mode == "constant":
        return _pad_constant(input, target_dims, pad_pairs, value)

    return _pad_non_constant(input, target_dims, pad_pairs, mode)


def pad_or_crop_to_size(
    input: Tensor,
    size: Sequence[int],
    mode: str = "constant",
    value: float = 0.0,
    dims: Sequence[int] | None = None,
) -> Tensor:
    """
    Pad or crop tensor to target size, centering the original content.

    Parameters
    ----------
    input
        Input tensor.
    size
        Target sizes for each dimension to adjust.
    mode
        'constant', 'circular', 'reflect', or 'replicate'.
    value
        Fill value for 'constant' mode.
    dims
        Dimensions to adjust. If None, adjusts last len(size) dimensions.

    Returns
    -------
        Tensor with specified dimensions resized to target.
    """
    ndim = input.ndim

    if dims is None:
        target_dims = list(range(ndim - len(size), ndim))
    else:
        if len(dims) != len(size):
            raise ValueError("len(dims) must equal len(size)")
        target_dims = [d % ndim for d in dims]

    pad_list: list[int] = []
    for dim, target in zip(target_dims, size):
        diff = target - input.shape[dim]
        pad_list.extend([diff // 2, diff - diff // 2])

    return pad_nd(input, pad_list, mode=mode, value=value, dims=target_dims)


def _pad_constant(
    input: Tensor,
    target_dims: list[int],
    pad_pairs: list[tuple[int, int]],
    value: float,
) -> Tensor:
    dim_to_pad = dict(zip(target_dims, pad_pairs))

    full_pad: list[int] = []
    for d in range(input.ndim - 1, -1, -1):
        full_pad.extend(dim_to_pad.get(d, (0, 0)))

    while len(full_pad) > 2 and full_pad[-2:] == [0, 0]:
        full_pad = full_pad[:-2]

    return torch.nn.functional.pad(input, full_pad, mode="constant", value=value)


def _pad_non_constant(
    input: Tensor,
    target_dims: list[int],
    pad_pairs: list[tuple[int, int]],
    mode: str,
) -> Tensor:
    result = input

    for i in range(0, len(target_dims), 3):
        chunk_dims = target_dims[i : i + 3]
        chunk_pads = pad_pairs[i : i + 3]

        other_dims = [d for d in range(result.ndim) if d not in chunk_dims]
        perm = other_dims + chunk_dims

        inv_perm = [0] * len(perm)
        for old_pos, new_pos in enumerate(perm):
            inv_perm[new_pos] = old_pos

        result = result.permute(perm)
        batch_shape = result.shape[: len(other_dims)]

        if len(batch_shape) == 0:
            result = result.unsqueeze(0)
        elif len(batch_shape) > 1:
            result = result.flatten(0, len(batch_shape) - 1)

        n_unsqueeze = max(0, len(chunk_dims) + 2 - result.ndim)
        for _ in range(n_unsqueeze):
            result = result.unsqueeze(1)

        flat_pad = [p for lr in reversed(chunk_pads) for p in lr]
        result = torch.nn.functional.pad(result, flat_pad, mode=mode)

        for _ in range(n_unsqueeze):
            result = result.squeeze(1)

        if len(batch_shape) == 0:
            result = result.squeeze(0)
        elif len(batch_shape) > 1:
            result = result.unflatten(0, batch_shape)

        result = result.permute(inv_perm)

    return result


def adjoint_pad_nd(
    input: Tensor,
    pad: Sequence[int],
    mode: str = "constant",
    value: float = 0.0,
    dims: Sequence[int] | None = None,
    original_sizes: Sequence[int] | None = None,
) -> Tensor:
    """
    Adjoint of pad_nd via autograd - computes backward of padding as forward op.

    Parameters
    ----------
    input
        Padded tensor (output space of pad_nd).
    pad
        Padding sizes as pairs (left, right) per dimension (same as pad_nd).
    mode
        'constant', 'circular', 'reflect', or 'replicate'.
    value
        Fill value for 'constant' mode.
    dims
        Dimensions that were padded. If None, assumes last N dimensions.
    original_sizes
        Original sizes of the padded dimensions before padding.
        Required to reconstruct the unpadded shape.

    Returns
    -------
        Tensor with padding adjoint applied (unpadded shape).
    """
    if len(pad) % 2 != 0:
        raise ValueError("pad must have even length")

    n_dims = len(pad) // 2
    if n_dims == 0:
        return input

    ndim = input.ndim

    if dims is None:
        target_dims = list(range(ndim - 1, ndim - n_dims - 1, -1))
    else:
        if len(dims) != n_dims:
            raise ValueError("len(dims) must equal len(pad) // 2")
        target_dims = [d % ndim for d in dims]

    if original_sizes is None:
        original_sizes = []
        for i, d in enumerate(target_dims):
            left, right = pad[2 * i], pad[2 * i + 1]
            original_sizes.append(input.shape[d] - left - right)

    unpadded_shape = list(input.shape)
    for i, d in enumerate(target_dims):
        unpadded_shape[d] = original_sizes[i]

    x = torch.zeros(unpadded_shape, device=input.device, dtype=input.dtype, requires_grad=True)
    y = pad_nd(x, pad, mode=mode, value=value, dims=dims)
    (gx,) = torch.autograd.grad(y, x, grad_outputs=input, retain_graph=False, create_graph=True)
    return gx
