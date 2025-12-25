"""Tests for ConvNd and ConvTransposeNd modules."""

import pytest
import torch
import torch.nn as nn

from torchnd import ConvNd, ConvTransposeNd, conv_nd


def make_shapes(ndim: int) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return (x_shape, w_shape, dim) for given spatial dimensionality."""
    x_shape = (2, 4) + (16,) * ndim
    w_shape = (8, 4) + (3,) * ndim
    dim = tuple(range(-ndim, 0))
    return x_shape, w_shape, dim


class TestForwardParity:
    """Verify module output matches functional implementation."""

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    @pytest.mark.parametrize("transposed", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
    def test_matches_functional(self, ndim: int, transposed: bool, dtype: torch.dtype) -> None:
        """Module forward pass equals conv_nd functional plus bias."""
        x_shape, _, dim = make_shapes(ndim)
        x = torch.randn(x_shape, dtype=dtype)

        if transposed:
            cls: type[ConvNd] | type[ConvTransposeNd] = ConvTransposeNd
            in_ch, out_ch = x_shape[1], 8
        else:
            cls = ConvNd
            in_ch, out_ch = x_shape[1], 8

        module = cls(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            dim=dim,
            padding=1,
            dtype=dtype,
            bias=True,
        )

        y_module = module(x)

        y_ref = conv_nd(x, module.weight, dim=dim, padding=1, transposed=transposed)
        bias_shape = [1] * y_ref.ndim
        bias_shape[1] = -1
        y_ref = y_ref + module.bias.view(*bias_shape)

        torch.testing.assert_close(y_module, y_ref)


class TestExternalWeights:
    """Test external weight/bias tensor handling."""

    @pytest.mark.parametrize("groups", [1, 2])
    @pytest.mark.parametrize("use_external", [False, True])
    def test_gradient_flow(self, groups: int, use_external: bool) -> None:
        """Gradients flow through external weights correctly."""
        in_ch, out_ch, kernel = 8, 16, 3
        w_shape = (out_ch, in_ch // groups, kernel, kernel)

        weight = torch.randn(w_shape, requires_grad=True) if use_external else None

        conv = ConvNd(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel,
            groups=groups,
            weight=weight,
            bias=False,
        )

        x = torch.randn(2, in_ch, 16, 16)
        conv(x).sum().backward()

        if use_external:
            assert not isinstance(conv.weight, nn.Parameter)
            assert weight is not None and weight.grad is not None
        else:
            assert isinstance(conv.weight, nn.Parameter)
            assert conv.weight.grad is not None

    def test_in_state_dict(self) -> None:
        """External weight is included in state_dict."""
        weight = torch.randn(8, 4, 3, 3)
        conv = ConvNd(4, 8, kernel_size=3, weight=weight, bias=False)

        state = conv.state_dict()
        assert "weight" in state
        torch.testing.assert_close(state["weight"], weight)

    def test_hypernetwork_pattern(self) -> None:
        """Gradient flows through hypernetwork producing weights."""
        hyper = nn.Linear(10, 8 * 4 * 9)
        z = torch.randn(10, requires_grad=True)
        weight = hyper(z).view(8, 4, 3, 3)

        conv = ConvNd(4, 8, kernel_size=3, weight=weight, bias=False)
        conv(torch.randn(2, 4, 16, 16)).sum().backward()

        assert z.grad is not None
        assert hyper.weight.grad is not None

    def test_external_bias_tensor(self) -> None:
        """External bias tensor applied correctly."""
        bias = torch.randn(8)
        conv = ConvNd(4, 8, kernel_size=3, bias=bias)

        assert not isinstance(conv.bias, nn.Parameter)
        assert "bias" in conv.state_dict()

        x = torch.randn(2, 4, 16, 16)
        expected = conv_nd(x, conv.weight, dim=(-2, -1)) + bias.view(1, -1, 1, 1)
        torch.testing.assert_close(conv(x), expected)


class TestAdjoint:
    """Test adjoint operator correctness."""

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
    @pytest.mark.parametrize(
        "config",
        [
            {"stride": 1, "padding": 0, "dilation": 1},
            {"stride": 2, "padding": 1, "dilation": 1},
            {"stride": 1, "padding": 2, "dilation": 2},
        ],
    )
    def test_inner_product_identity_conv(self, ndim: int, dtype: torch.dtype, config: dict[str, int]) -> None:
        """Verify <Ax, y> == <x, A^H y> for ConvNd."""
        x_shape, _, dim = make_shapes(ndim)
        in_ch, out_ch = 4, 8

        conv = ConvNd(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            dim=dim,
            bias=False,
            dtype=dtype,
            **config,  # type: ignore[arg-type]
        )

        input_shape = (x_shape[0], in_ch, *x_shape[2:])
        x = torch.randn(input_shape, dtype=dtype)
        y = torch.randn_like(conv(x))

        lhs = torch.vdot(conv(x).flatten(), y.flatten())
        rhs = torch.vdot(x.flatten(), conv.adjoint(y, input_shape=input_shape[2:]).flatten())

        rtol = 1e-4 if dtype == torch.float32 else 1e-5
        torch.testing.assert_close(lhs, rhs, rtol=rtol, atol=rtol)

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
    @pytest.mark.parametrize(
        "config",
        [
            {"stride": 1, "padding": 0, "dilation": 1},
            {"stride": 2, "padding": 1, "dilation": 1},
            {"stride": 1, "padding": 2, "dilation": 2},
        ],
    )
    def test_inner_product_identity_transposed(self, ndim: int, dtype: torch.dtype, config: dict[str, int]) -> None:
        """Verify <Ax, y> == <x, A^H y> for ConvTransposeNd."""
        x_shape, _, dim = make_shapes(ndim)
        in_ch, out_ch = 8, 4

        conv = ConvTransposeNd(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            dim=dim,
            bias=False,
            dtype=dtype,
            **config,  # type: ignore[arg-type]
        )

        spatial_in = tuple(s // 2 for s in x_shape[2:])
        input_shape = (x_shape[0], in_ch, *spatial_in)

        x = torch.randn(input_shape, dtype=dtype)
        y = torch.randn_like(conv(x))

        lhs = torch.vdot(conv(x).flatten(), y.flatten())
        rhs = torch.vdot(x.flatten(), conv.adjoint(y).flatten())

        rtol = 1e-4 if dtype == torch.float32 else 1e-5
        torch.testing.assert_close(lhs, rhs, rtol=rtol, atol=rtol)

    def test_raises_with_bias(self) -> None:
        """Adjoint raises ValueError if bias is present."""
        conv = ConvNd(4, 8, kernel_size=3, bias=True)
        with pytest.raises(ValueError, match="bias"):
            conv.adjoint(torch.randn(4, 8, 16, 16))


class TestDeviceDtype:
    """Test device and dtype handling."""

    def test_external_weight_dtype_cast(self) -> None:
        """External weight is cast to specified dtype."""
        w = torch.randn(8, 4, 3, 3, dtype=torch.float32)
        conv = ConvNd(4, 8, kernel_size=3, weight=w, bias=False, dtype=torch.float64)
        assert conv.weight.dtype == torch.float64

    def test_complex_dtype_initialization(self) -> None:
        """Complex dtype initializes both real and imaginary parts."""
        conv = ConvNd(4, 8, kernel_size=3, dtype=torch.complex64)

        assert conv.weight.is_complex()
        assert conv.bias.is_complex()
        assert conv.weight.real.std() > 0
        assert conv.weight.imag.std() > 0

    def test_module_to_device(self) -> None:
        """Module.to() moves all parameters including buffers."""
        weight = torch.randn(8, 4, 3, 3)
        conv = ConvNd(4, 8, kernel_size=3, weight=weight, bias=True)
        conv = conv.to("meta")

        assert conv.weight.device.type == "meta"
        assert conv.bias.device.type == "meta"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    def test_external_weight_device_move(self) -> None:
        """External weight moved to specified device."""
        w = torch.randn(8, 4, 3, 3, device="cpu")
        conv = ConvNd(4, 8, kernel_size=3, weight=w, bias=False, device=torch.device("cuda"))
        assert conv.weight.device.type == "cuda"


class TestDimConfiguration:
    """Test flexible dimension specification."""

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    def test_spatial_dims(self, ndim: int) -> None:
        """Convolution works for 1D, 2D, 3D spatial dims."""
        x_shape, _, dim = make_shapes(ndim)
        conv = ConvNd(4, 8, kernel_size=3, dim=dim, padding=1)
        x = torch.randn(x_shape)
        y = conv(x)
        assert y.shape == (2, 8) + x_shape[2:]

    def test_custom_channel_dim(self) -> None:
        """Channel dimension can be non-standard position."""
        conv = ConvNd(4, 8, kernel_size=3, dim=(1, 2), channel_dim=-1, padding=1)
        x = torch.randn(2, 16, 16, 4)
        y = conv(x)
        assert y.shape == (2, 16, 16, 8)

    def test_extra_batch_dims(self) -> None:
        """Multiple batch dimensions handled correctly."""
        conv = ConvNd(4, 8, kernel_size=3, dim=(-2, -1), channel_dim=2, padding=1)
        x = torch.randn(2, 3, 4, 16, 16)
        y = conv(x)
        assert y.shape == (2, 3, 8, 16, 16)


class TestGroups:
    """Test grouped convolution."""

    @pytest.mark.parametrize("transposed", [False, True])
    def test_grouped_weight_shape(self, transposed: bool) -> None:
        """Weight shape accounts for groups."""
        cls = ConvTransposeNd if transposed else ConvNd
        conv = cls(8, 16, kernel_size=3, groups=2)

        if transposed:
            assert conv.weight.shape == (8, 8, 3, 3)
        else:
            assert conv.weight.shape == (16, 4, 3, 3)

    def test_invalid_in_channels(self) -> None:
        """Raises if in_channels not divisible by groups."""
        with pytest.raises(ValueError, match="in_channels"):
            ConvNd(7, 8, kernel_size=3, groups=2)

    def test_invalid_out_channels(self) -> None:
        """Raises if out_channels not divisible by groups."""
        with pytest.raises(ValueError, match="out_channels"):
            ConvNd(8, 7, kernel_size=3, groups=2)


class TestResetParameters:
    """Test parameter initialization."""

    def test_reinitializes_learnable(self) -> None:
        """reset_parameters reinitializes learnable weights."""
        conv = ConvNd(4, 8, kernel_size=3)
        conv.weight.data.zero_()
        conv.reset_parameters()
        assert conv.weight.abs().sum() > 0

    def test_skips_external(self) -> None:
        """reset_parameters does not modify external weights."""
        weight = torch.ones(8, 4, 3, 3)
        conv = ConvNd(4, 8, kernel_size=3, weight=weight, bias=False)
        conv.reset_parameters()
        torch.testing.assert_close(conv.weight, torch.ones(8, 4, 3, 3))


class TestStateDictRoundtrip:
    """Test state_dict save/load."""

    @pytest.mark.parametrize("use_external", [False, True])
    def test_roundtrip(self, use_external: bool) -> None:
        """State dict loads correctly for both learnable and external weights."""
        weight = torch.randn(8, 4, 3, 3) if use_external else None
        conv1 = ConvNd(4, 8, kernel_size=3, weight=weight, bias=True)

        weight2 = torch.zeros(8, 4, 3, 3) if use_external else None
        conv2 = ConvNd(4, 8, kernel_size=3, weight=weight2, bias=True)

        conv2.load_state_dict(conv1.state_dict())

        torch.testing.assert_close(conv2.weight, conv1.weight)
        torch.testing.assert_close(conv2.bias, conv1.bias)


class TestAutograd:
    """Test gradient computation."""

    def test_gradcheck(self) -> None:
        """Numerical gradient check passes."""
        conv = ConvNd(2, 3, kernel_size=3, padding=1, bias=True, dtype=torch.float64)
        x = torch.randn(1, 2, 8, 8, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(conv, x)

    def test_backward_all_params(self) -> None:
        """Backward computes gradients for input, weight, and bias."""
        conv = ConvNd(4, 8, kernel_size=3, padding=1)
        x = torch.randn(2, 4, 16, 16, requires_grad=True)
        conv(x).sum().backward()

        assert x.grad is not None
        assert conv.weight.grad is not None
        assert conv.bias.grad is not None


class TestRepr:
    """Test string representation."""

    def test_contains_essential_info(self) -> None:
        """Repr contains channels, kernel, stride."""
        conv = ConvNd(4, 8, kernel_size=3, stride=2, padding=1)
        r = repr(conv)
        assert "4" in r and "8" in r
        assert "stride" in r

    def test_shows_external(self) -> None:
        """Repr indicates external weight/bias."""
        weight = torch.randn(8, 4, 3, 3)
        bias = torch.randn(8)
        conv = ConvNd(4, 8, kernel_size=3, weight=weight, bias=bias)
        r = repr(conv)
        assert "external" in r.lower()

    def test_shows_padding_mode(self) -> None:
        """Repr shows padding_mode when not zeros."""
        conv = ConvNd(4, 8, kernel_size=3, padding=1, padding_mode="reflect")
        r = repr(conv)
        assert "reflect" in r


class TestPaddingMode:
    """Test padding_mode support for different modes and per-dimension configuration."""

    @pytest.mark.parametrize("mode", ["zeros", "reflect", "replicate", "circular"])
    def test_convnd_single_mode(self, mode: str) -> None:
        """ConvNd works with each padding mode."""
        conv = ConvNd(4, 8, kernel_size=3, padding=2, padding_mode=mode)
        x = torch.randn(2, 4, 16, 16)
        y = conv(x)
        assert y.shape == (2, 8, 18, 18)

    @pytest.mark.parametrize("mode", ["zeros", "reflect", "replicate", "circular"])
    def test_transpose_single_mode(self, mode: str) -> None:
        """ConvTransposeNd works with each padding mode."""
        conv = ConvTransposeNd(8, 4, kernel_size=3, padding=2, padding_mode=mode)
        x = torch.randn(2, 8, 8, 8)
        y = conv(x)
        # With padding=2 and kernel=3, output is: (8-1)*1 - 2*2 + 3 = 6
        assert y.shape == (2, 4, 6, 6)

    def test_convnd_tuple_mode(self) -> None:
        """ConvNd accepts per-dimension padding modes."""
        conv = ConvNd(4, 8, kernel_size=3, padding=2, padding_mode=("reflect", "circular"))
        x = torch.randn(2, 4, 16, 16)
        y = conv(x)
        assert y.shape == (2, 8, 18, 18)

    def test_transpose_tuple_mode(self) -> None:
        """ConvTransposeNd accepts per-dimension padding modes."""
        conv = ConvTransposeNd(8, 4, kernel_size=3, padding=2, padding_mode=("reflect", "replicate"))
        x = torch.randn(2, 8, 8, 8)
        y = conv(x)
        assert y.shape == (2, 4, 6, 6)

    def test_mixed_zeros_and_nonzeros(self) -> None:
        """Mixed padding modes work correctly."""
        conv = ConvNd(4, 8, kernel_size=3, padding=2, padding_mode=("zeros", "reflect"))
        x = torch.randn(2, 4, 16, 16)
        y = conv(x)
        assert y.shape == (2, 8, 18, 18)

    @pytest.mark.parametrize("mode", ["reflect", "replicate", "circular"])
    def test_gradient_flow_convnd(self, mode: str) -> None:
        """Gradients flow correctly through non-zeros padding modes in ConvNd."""
        conv = ConvNd(4, 8, kernel_size=3, padding=2, padding_mode=mode, bias=True)
        x = torch.randn(2, 4, 16, 16, requires_grad=True)
        conv(x).sum().backward()

        assert x.grad is not None
        assert conv.weight.grad is not None
        assert conv.bias.grad is not None

    @pytest.mark.parametrize("mode", ["reflect", "replicate", "circular"])
    def test_gradient_flow_transpose(self, mode: str) -> None:
        """Gradients flow correctly through non-zeros padding modes in ConvTransposeNd."""
        conv = ConvTransposeNd(8, 4, kernel_size=3, padding=2, padding_mode=mode, bias=True)
        x = torch.randn(2, 8, 8, 8, requires_grad=True)
        conv(x).sum().backward()

        assert x.grad is not None
        assert conv.weight.grad is not None
        assert conv.bias.grad is not None

    def test_invalid_mode_raises(self) -> None:
        """Invalid padding mode raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            ConvNd(4, 8, kernel_size=3, padding=1, padding_mode="invalid")

    def test_invalid_tuple_mode_raises(self) -> None:
        """Invalid mode in tuple raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            ConvNd(4, 8, kernel_size=3, padding=1, padding_mode=("reflect", "invalid"))


class TestPaddingModeAdjoint:
    """Test adjoint correctness with non-zeros padding modes."""

    @pytest.mark.parametrize("mode", ["zeros", "reflect", "replicate", "circular"])
    @pytest.mark.parametrize("ndim", [1, 2])
    def test_inner_product_identity_conv(self, mode: str, ndim: int) -> None:
        """Verify <Ax, y> == <x, A^H y> for ConvNd with padding_mode."""
        x_shape, _, dim = make_shapes(ndim)
        in_ch, out_ch = 4, 8

        conv = ConvNd(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            dim=dim,
            padding=2,
            padding_mode=mode,
            bias=False,
        )

        input_shape = (x_shape[0], in_ch, *x_shape[2:])
        x = torch.randn(input_shape)
        y = torch.randn_like(conv(x))

        lhs = torch.vdot(conv(x).flatten(), y.flatten())
        rhs = torch.vdot(x.flatten(), conv.adjoint(y, input_shape=input_shape[2:]).flatten())

        torch.testing.assert_close(lhs, rhs, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("mode", ["zeros", "reflect", "replicate", "circular"])
    @pytest.mark.parametrize("ndim", [1, 2])
    def test_inner_product_identity_transpose(self, mode: str, ndim: int) -> None:
        """Verify <Ax, y> == <x, A^H y> for ConvTransposeNd with padding_mode."""
        x_shape, _, dim = make_shapes(ndim)
        in_ch, out_ch = 8, 4

        conv = ConvTransposeNd(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            dim=dim,
            padding=2,
            padding_mode=mode,
            bias=False,
        )

        spatial_in = tuple(s // 2 for s in x_shape[2:])
        input_shape = (x_shape[0], in_ch, *spatial_in)

        x = torch.randn(input_shape)
        y = torch.randn_like(conv(x))

        lhs = torch.vdot(conv(x).flatten(), y.flatten())
        rhs = torch.vdot(x.flatten(), conv.adjoint(y).flatten())

        torch.testing.assert_close(lhs, rhs, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize(
        "modes",
        [
            ("reflect", "circular"),
            ("zeros", "reflect"),
            ("replicate", "zeros"),
            ("circular", "replicate"),
        ],
    )
    def test_mixed_modes_conv_adjoint(self, modes: tuple[str, str]) -> None:
        """Verify adjoint identity with mixed per-dimension padding modes for ConvNd."""
        conv = ConvNd(
            in_channels=4,
            out_channels=8,
            kernel_size=3,
            dim=(-2, -1),
            padding=2,
            padding_mode=modes,
            bias=False,
        )

        x = torch.randn(2, 4, 16, 16)
        y = torch.randn_like(conv(x))

        lhs = torch.vdot(conv(x).flatten(), y.flatten())
        rhs = torch.vdot(x.flatten(), conv.adjoint(y, input_shape=(16, 16)).flatten())

        torch.testing.assert_close(lhs, rhs, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize(
        "modes",
        [
            ("reflect", "circular"),
            ("zeros", "reflect"),
            ("replicate", "zeros"),
            ("circular", "replicate"),
        ],
    )
    def test_mixed_modes_transpose_adjoint(self, modes: tuple[str, str]) -> None:
        """Verify adjoint identity with mixed per-dimension padding modes for ConvTransposeNd."""
        conv = ConvTransposeNd(
            in_channels=8,
            out_channels=4,
            kernel_size=3,
            dim=(-2, -1),
            padding=2,
            padding_mode=modes,
            bias=False,
        )

        x = torch.randn(2, 8, 8, 8)
        y = torch.randn_like(conv(x))

        lhs = torch.vdot(conv(x).flatten(), y.flatten())
        rhs = torch.vdot(x.flatten(), conv.adjoint(y).flatten())

        torch.testing.assert_close(lhs, rhs, rtol=1e-4, atol=1e-5)

    def test_3d_mixed_modes_adjoint(self) -> None:
        """Verify adjoint identity with 3 different padding modes."""
        conv = ConvNd(
            in_channels=4,
            out_channels=8,
            kernel_size=3,
            dim=(-3, -2, -1),
            padding=1,
            padding_mode=("reflect", "replicate", "circular"),
            bias=False,
        )

        x = torch.randn(2, 4, 8, 8, 8)
        y = torch.randn_like(conv(x))

        lhs = torch.vdot(conv(x).flatten(), y.flatten())
        rhs = torch.vdot(x.flatten(), conv.adjoint(y, input_shape=(8, 8, 8)).flatten())

        torch.testing.assert_close(lhs, rhs, rtol=1e-4, atol=1e-5)
