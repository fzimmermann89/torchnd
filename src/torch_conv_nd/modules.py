"""N-dimensional convolution modules wrapping conv_nd functional interface."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .functional import _normalize_tuple, adjoint_pad_nd, conv_nd, pad_nd


class _ConvNdBase(torch.nn.Module, ABC):
    """Base class for N-dimensional convolution modules."""

    transposed: bool = False
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, ...]
    dim: tuple[int, ...]
    channel_dim: int
    stride: tuple[int, ...]
    padding: tuple[int, ...]
    output_padding: tuple[int, ...]
    dilation: tuple[int, ...]
    groups: int
    padding_mode: tuple[str, ...]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        dim: tuple[int, ...] = (-2, -1),
        channel_dim: int = 1,
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
        output_padding: int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool | Tensor = True,
        padding_mode: str | tuple[str, ...] = "zeros",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        weight: Tensor | None = None,
    ) -> None:
        """Initialize N-dimensional convolution.

        Parameters
        ----------
        in_channels
            Number of input channels.
        out_channels
            Number of output channels.
        kernel_size
            Size of the convolving kernel.
        dim
            Spatial dimensions to convolve over.
        channel_dim
            Channel dimension index.
        stride
            Stride of the convolution.
        padding
            Padding added to each spatial dimension.
        output_padding
            Additional size added to output shape (transposed conv only).
        dilation
            Spacing between kernel elements.
        groups
            Number of blocked connections from input to output channels.
        bias
            If True, adds learnable bias. If Tensor, uses as external bias.
        padding_mode
            Supported padding modes are 'zeros', 'reflect', 'circular', and 'replicate'.
            Can be a single string (applied to all dims) or a tuple of strings per dim.
        device
            Device for parameters. External weights/bias are moved to this device.
        dtype
            Data type for parameters. Complex dtypes enable complex convolution.
            External weights/bias are cast to this dtype.
        weight
            If provided, uses as external weight (enables hypernetwork usage).
            Registered as buffer to be included in state_dict.
        """
        super().__init__()

        if in_channels % groups != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by groups ({groups})")
        if out_channels % groups != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by groups ({groups})")

        num_spatial = len(dim)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _normalize_tuple(kernel_size, num_spatial)
        self.dim = dim
        self.channel_dim = channel_dim
        self.stride = _normalize_tuple(stride, num_spatial)
        self.padding = _normalize_tuple(padding, num_spatial)
        self.output_padding = _normalize_tuple(output_padding, num_spatial)
        self.dilation = _normalize_tuple(dilation, num_spatial)
        self.groups = groups

        if isinstance(padding_mode, str):
            self.padding_mode = (padding_mode,) * num_spatial
        else:
            self.padding_mode = tuple(padding_mode)

        if self.transposed:
            weight_shape = (in_channels, out_channels // groups, *self.kernel_size)
        else:
            weight_shape = (out_channels, in_channels // groups, *self.kernel_size)

        for mode in self.padding_mode:
            if mode not in ("zeros", "reflect", "circular", "replicate"):
                raise ValueError(f"padding_mode '{mode}' is not supported")

        if weight is not None:
            weight = self._apply_device_dtype(weight, device, dtype)
            self.register_buffer("weight", weight)
        else:
            self.weight = torch.nn.Parameter(torch.empty(weight_shape, device=device, dtype=dtype))

        if isinstance(bias, Tensor):
            bias = self._apply_device_dtype(bias, device, dtype)
            self.register_buffer("bias", bias)
        elif bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels, device=device, dtype=dtype))
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()

    @staticmethod
    def _apply_device_dtype(
        tensor: Tensor,
        device: torch.device | None,
        dtype: torch.dtype | None,
    ) -> Tensor:
        if device is not None or dtype is not None:
            tensor = tensor.to(device=device, dtype=dtype)
        return tensor

    def reset_parameters(self) -> None:
        """Initialize learnable parameters using Kaiming uniform initialization."""
        if not isinstance(self.weight, torch.nn.Parameter):
            return

        fan_in = self.weight.shape[1] * math.prod(self.weight.shape[2:])

        if self.weight.is_complex():
            std = 1.0 / math.sqrt(fan_in * 2)
            bound = math.sqrt(3.0) * std
            with torch.no_grad():
                self.weight.real.uniform_(-bound, bound)
                self.weight.imag.uniform_(-bound, bound)
        else:
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if isinstance(self.bias, torch.nn.Parameter):
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
            with torch.no_grad():
                if self.bias.is_complex():
                    self.bias.real.uniform_(-bound, bound)
                    self.bias.imag.uniform_(-bound, bound)
                else:
                    self.bias.uniform_(-bound, bound)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the convolution."""
        raise NotImplementedError

    @abstractmethod
    def adjoint(self, x: Tensor) -> Tensor:  # noqa: ARG002
        """Return the adjoint convolution.

        The adjoint of a convolution is the transposed convolution with
        conjugated weights.

        Parameters
        ----------
        x
            Input tensor.
        Returns
        -------
            Adjoint convolution.
        """
        raise NotImplementedError

    def extra_repr(self) -> str:
        parts = [
            f"{self.in_channels}",
            f"{self.out_channels}",
            f"kernel_size={self.kernel_size}",
            f"stride={self.stride}",
        ]
        if self.padding != (0,) * len(self.padding):
            parts.append(f"padding={self.padding}")
        if self.dilation != (1,) * len(self.dilation):
            parts.append(f"dilation={self.dilation}")
        if self.transposed and self.output_padding != (0,) * len(self.output_padding):
            parts.append(f"output_padding={self.output_padding}")
        if self.groups != 1:
            parts.append(f"groups={self.groups}")
        if self.bias is None:
            parts.append("bias=False")
        if not all(m == "zeros" for m in self.padding_mode):
            # Show single value if all same, otherwise show tuple
            if len(set(self.padding_mode)) == 1:
                parts.append(f"padding_mode={self.padding_mode[0]!r}")
            else:
                parts.append(f"padding_mode={self.padding_mode}")
        if self.dim != (-2, -1):
            parts.append(f"dim={self.dim}")
        if self.channel_dim != 1:
            parts.append(f"channel_dim={self.channel_dim}")
        if not isinstance(self.weight, torch.nn.Parameter):
            parts.append("weight=external")
        if self.bias is not None and not isinstance(self.bias, torch.nn.Parameter):
            parts.append("bias=external")
        return ", ".join(parts)


class ConvNd(_ConvNdBase):
    """N-dimensional convolution with flexible dimension specification.

    Supports arbitrary spatial dimensions and dimension layouts. When native
    PyTorch implementations (1D, 2D, 3D) are unavailable, uses recursive
    decomposition.

    Examples
    --------
    >>> conv = ConvNd(3, 64, kernel_size=3, dim=(-2, -1))
    >>> x = torch.randn(2, 3, 32, 32)
    >>> y = conv(x)

    >>> # Hypernetwork usage
    >>> external_weight = hyper_net(z)
    >>> conv = ConvNd(3, 64, kernel_size=3, weight=external_weight, bias=False)
    >>> y = conv(x)  # Gradients flow through hyper_net
    """

    transposed = False

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        dim: tuple[int, ...] = (-2, -1),
        channel_dim: int = 1,
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool | Tensor = True,
        padding_mode: str | tuple[str, ...] = "zeros",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        weight: Tensor | None = None,
    ) -> None:
        """Initialize N-dimensional convolution.

        Parameters
        ----------
        in_channels
            Number of input channels.
        out_channels
            Number of output channels.
        kernel_size
            Size of the convolving kernel.
        dim
            Spatial dimensions to convolve over.
        channel_dim
            Channel dimension index.
        stride
            Stride of the convolution.
        padding
            Padding added to each spatial dimension.
        dilation
            Spacing between kernel elements.
        groups
            Number of blocked connections from input to output channels.
        bias
            If True, adds learnable bias. If Tensor, uses as external bias.
        padding_mode
            Supported padding modes are 'zeros', 'reflect', 'circular', and 'replicate'.
            Can be a single string (applied to all dims) or a tuple of strings per dim.
        device
            Device for parameters. External weights/bias are moved to this device.
        dtype
            Data type for parameters. Complex dtypes enable complex convolution.
        weight
            If provided, uses as external weight. Registered as buffer for state_dict.
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dim=dim,
            channel_dim=channel_dim,
            stride=stride,
            padding=padding,
            output_padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
            weight=weight,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the forward convolution operator.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
            Output tensor.
        """
        effective_padding = self.padding

        if not all(m == "zeros" for m in self.padding_mode):
            effective_padding = tuple(0 if m != "zeros" and p > 0 else p for m, p in zip(self.padding_mode, self.padding))
            for mode in ("reflect", "replicate", "circular"):
                dims_idx = [i for i, (m, p) in enumerate(zip(self.padding_mode, self.padding)) if m == mode and p > 0]
                if dims_idx:
                    spatial_dims = tuple(self.dim[i] for i in dims_idx)
                    pad_spec = [v for i in dims_idx for v in (self.padding[i], self.padding[i])]
                    x = checkpoint(pad_nd, x, pad_spec, mode, 0.0, spatial_dims, use_reentrant=False)

        out = conv_nd(
            x,
            self.weight,
            dim=self.dim,
            channel_dim=self.channel_dim,
            stride=self.stride,
            padding=effective_padding,
            dilation=self.dilation,
            groups=self.groups,
            transposed=False,
        )

        if self.bias is not None:
            bias_shape = [1] * out.ndim
            bias_shape[self.channel_dim % out.ndim] = -1
            out = out + self.bias.view(*bias_shape)

        return out

    def adjoint(self, x: Tensor, input_shape: tuple[int, ...] | None = None) -> Tensor:
        """Apply the adjoint convolution operator.

        The adjoint of a convolution is the transposed convolution with
        conjugated weights (for complex) or identical weights (for real).

        This method applies the adjoint using the same weights (and does not
        allocate a new module). Intended for iterative algorithms, not separate
        training of an adjoint operator.

        Parameters
        ----------
        x
            Input tensor in the output space of the forward operator.
        input_shape
            Spatial shape of the input to the forward operator. Needed for
            strided ConvNd to determine output_padding of the adjoint. For
            stride > 1, multiple input shapes can map to the same output shape,
            so the adjoint output size is ambiguous without this information.

        Returns
        -------
            Result of applying the adjoint operator.

        Raises
        ------
        ValueError
            If bias is present (adjoint undefined with bias).
            If input_shape is missing when any stride != 1.
        """
        if self.bias is not None:
            raise ValueError("Adjoint is not defined for convolution with bias")

        effective_padding = self.padding
        if not all(m == "zeros" for m in self.padding_mode):
            effective_padding = tuple(0 if m != "zeros" and p > 0 else p for m, p in zip(self.padding_mode, self.padding))

        num_spatial = len(self.dim)
        if input_shape is None:
            if any(s != 1 for s in self.stride):
                raise ValueError("input_shape is required for ConvNd.adjoint when any stride != 1")
            out_pad = (0,) * num_spatial
        else:
            if len(input_shape) != num_spatial:
                raise ValueError(f"input_shape must have length {num_spatial}, got {len(input_shape)}")

            pads: list[int] = []
            for i, (n_in, k, s, p, d) in enumerate(zip(input_shape, self.kernel_size, self.stride, self.padding, self.dilation)):
                eff_p = effective_padding[i]
                n_out = (n_in + 2 * p - d * (k - 1) - 1) // s + 1
                base = (n_out - 1) * s - 2 * eff_p + d * (k - 1) + 1
                op = (n_in + 2 * (p - eff_p)) - base
                if op < 0 or op >= s:
                    raise RuntimeError(
                        "Computed output_padding is invalid. "
                        f"Got output_padding={op} for stride={s} "
                        f"(input={n_in}, kernel={k}, pad={p}, dilation={d})."
                    )
                pads.append(op)
            out_pad = tuple(pads)

        w = self.weight.conj() if self.weight.is_complex() else self.weight
        out = conv_nd(
            x,
            w,
            dim=self.dim,
            channel_dim=self.channel_dim,
            stride=self.stride,
            padding=effective_padding,
            dilation=self.dilation,
            groups=self.groups,
            transposed=True,
            output_padding=out_pad,
        )

        if not all(m == "zeros" for m in self.padding_mode):
            for mode in ("reflect", "replicate", "circular"):
                dims_idx = [i for i, (m, p) in enumerate(zip(self.padding_mode, self.padding)) if m == mode and p > 0]
                if dims_idx:
                    spatial_dims = tuple(self.dim[i] for i in dims_idx)
                    pad_spec = [v for i in dims_idx for v in (self.padding[i], self.padding[i])]
                    original_sizes = [input_shape[i] for i in dims_idx] if input_shape else None
                    out = adjoint_pad_nd(out, pad_spec, mode=mode, dims=spatial_dims, original_sizes=original_sizes)

        return out


class ConvTransposeNd(_ConvNdBase):
    """N-dimensional transposed convolution with flexible dimension specification.

    Also known as fractionally-strided convolution or deconvolution.

    Examples
    --------
    >>> conv_t = ConvTransposeNd(64, 3, kernel_size=3, stride=2, dim=(-2, -1))
    >>> x = torch.randn(2, 64, 16, 16)
    >>> y = conv_t(x)
    """

    transposed = True

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        dim: tuple[int, ...] = (-2, -1),
        channel_dim: int = 1,
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
        output_padding: int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool | Tensor = True,
        padding_mode: str | tuple[str, ...] = "zeros",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        weight: Tensor | None = None,
    ) -> None:
        """Initialize N-dimensional transposed convolution.

        Parameters
        ----------
        in_channels
            Number of input channels.
        out_channels
            Number of output channels.
        kernel_size
            Size of the convolving kernel.
        dim
            Spatial dimensions to convolve over.
        channel_dim
            Channel dimension index.
        stride
            Stride of the convolution.
        padding
            Padding added to each spatial dimension.
        output_padding
            Additional size added to output shape to disambiguate when stride > 1.
        dilation
            Spacing between kernel elements.
        groups
            Number of blocked connections from input to output channels.
        bias
            If True, adds learnable bias. If Tensor, uses as external bias.
        padding_mode
            Supported padding modes are 'zeros', 'reflect', 'circular', and 'replicate'.
            Can be a single string (applied to all dims) or a tuple of strings per dim.
        device
            Device for parameters. External weights/bias are moved to this device.
        dtype
            Data type for parameters. Complex dtypes enable complex convolution.
        weight
            If provided, uses as external weight. Registered as buffer for state_dict.
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dim=dim,
            channel_dim=channel_dim,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
            weight=weight,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Apply the forward convolution operator.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
            Output tensor.
        """

        effective_padding = self.padding
        if not all(m == "zeros" for m in self.padding_mode):
            effective_padding = tuple(0 if m != "zeros" and p > 0 else p for m, p in zip(self.padding_mode, self.padding))

        out = conv_nd(
            x,
            self.weight,
            dim=self.dim,
            channel_dim=self.channel_dim,
            stride=self.stride,
            padding=effective_padding,
            dilation=self.dilation,
            groups=self.groups,
            transposed=True,
            output_padding=self.output_padding,
        )

        if not all(m == "zeros" for m in self.padding_mode):
            for mode in ("reflect", "replicate", "circular"):
                dims_idx = [i for i, (m, p) in enumerate(zip(self.padding_mode, self.padding)) if m == mode and p > 0]
                if dims_idx:
                    spatial_dims = tuple(self.dim[i] for i in dims_idx)
                    pad_spec = [v for i in dims_idx for v in (self.padding[i], self.padding[i])]
                    original_sizes = [out.shape[self.dim[i]] - 2 * self.padding[i] for i in dims_idx]
                    out = checkpoint(
                        adjoint_pad_nd,
                        out,
                        pad_spec,
                        mode,
                        0.0,
                        spatial_dims,
                        original_sizes,
                        use_reentrant=False,
                    )

        if self.bias is not None:
            bias_shape = [1] * out.ndim
            bias_shape[self.channel_dim % out.ndim] = -1
            out = out + self.bias.view(*bias_shape)

        return out

    def adjoint(self, x: Tensor) -> Tensor:
        """Apply the adjoint convolution operator.

        The adjoint of a transposed convolution is the (forward) convolution with
        conjugated weights (for complex) or identical weights (for real).

        This method applies the adjoint using the same weights (and does not
        allocate a new module). Intended for iterative algorithms, not separate
        training of an adjoint operator.

        Parameters
        ----------
        x
            Input tensor in the output space of the forward operator.

        Returns
        -------
            Result of applying the adjoint operator.

        Raises
        ------
        ValueError
            If bias is present (adjoint undefined with bias).
        """
        if self.bias is not None:
            raise ValueError("Adjoint is not defined for convolution with bias")

        effective_padding = self.padding
        if not all(m == "zeros" for m in self.padding_mode):
            effective_padding = tuple(0 if m != "zeros" and p > 0 else p for m, p in zip(self.padding_mode, self.padding))
            for mode in ("reflect", "replicate", "circular"):
                dims_idx = [i for i, (m, p) in enumerate(zip(self.padding_mode, self.padding)) if m == mode and p > 0]
                if dims_idx:
                    spatial_dims = tuple(self.dim[i] for i in dims_idx)
                    pad_spec = [v for i in dims_idx for v in (self.padding[i], self.padding[i])]
                    x = pad_nd(x, pad_spec, mode=mode, dims=spatial_dims)

        w = self.weight.conj() if self.weight.is_complex() else self.weight
        return conv_nd(
            x,
            w,
            dim=self.dim,
            channel_dim=self.channel_dim,
            stride=self.stride,
            padding=effective_padding,
            dilation=self.dilation,
            groups=self.groups,
            transposed=False,
        )
