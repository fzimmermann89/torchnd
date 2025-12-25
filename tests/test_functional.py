"""Tests for conv_nd."""

from collections.abc import Iterator

import pytest
import torch

from torchnd import adjoint_pad_nd, conv_nd, pad_nd, pad_or_crop_to_size
from torchnd.functional import CONV_REGISTRY, CONV_TRANSPOSE_REGISTRY


@pytest.fixture
def disable_conv3d() -> Iterator[None]:
    orig, orig_t = CONV_REGISTRY.copy(), CONV_TRANSPOSE_REGISTRY.copy()
    del CONV_REGISTRY[3], CONV_TRANSPOSE_REGISTRY[3]
    yield
    CONV_REGISTRY.clear()
    CONV_REGISTRY.update(orig)
    CONV_TRANSPOSE_REGISTRY.clear()
    CONV_TRANSPOSE_REGISTRY.update(orig_t)


@pytest.fixture
def disable_conv2d_3d() -> Iterator[None]:
    orig, orig_t = CONV_REGISTRY.copy(), CONV_TRANSPOSE_REGISTRY.copy()
    CONV_REGISTRY.pop(2, None)
    CONV_REGISTRY.pop(3, None)
    CONV_TRANSPOSE_REGISTRY.pop(2, None)
    CONV_TRANSPOSE_REGISTRY.pop(3, None)
    yield
    CONV_REGISTRY.clear()
    CONV_REGISTRY.update(orig)
    CONV_TRANSPOSE_REGISTRY.clear()
    CONV_TRANSPOSE_REGISTRY.update(orig_t)


class TestForwardNative:
    def test_conv1d(self) -> None:
        x, w = torch.randn(2, 4, 32), torch.randn(8, 4, 3)
        expected = torch.nn.functional.conv1d(x, w, stride=2, padding=1)
        result = conv_nd(x, w, dim=(-1,), stride=2, padding=1)
        torch.testing.assert_close(result, expected)

    def test_conv2d(self) -> None:
        x, w = torch.randn(2, 4, 16, 16), torch.randn(8, 4, 3, 3)
        expected = torch.nn.functional.conv2d(x, w, stride=2, padding=1, dilation=2)
        result = conv_nd(x, w, dim=(-2, -1), stride=2, padding=1, dilation=2)
        torch.testing.assert_close(result, expected)

    def test_conv3d(self) -> None:
        x, w = torch.randn(2, 4, 8, 8, 8), torch.randn(8, 4, 3, 3, 3)
        expected = torch.nn.functional.conv3d(x, w, stride=2, padding=1)
        result = conv_nd(x, w, dim=(-3, -2, -1), stride=2, padding=1)
        torch.testing.assert_close(result, expected)


class TestTransposeNative:
    def test_conv_transpose1d(self) -> None:
        x, w = torch.randn(2, 4, 8), torch.randn(4, 8, 3)
        expected = torch.nn.functional.conv_transpose1d(x, w, stride=2, padding=1)
        result = conv_nd(x, w, dim=(-1,), stride=2, padding=1, transposed=True)
        torch.testing.assert_close(result, expected)

    def test_conv_transpose2d(self) -> None:
        x, w = torch.randn(2, 4, 8, 8), torch.randn(4, 8, 3, 3)
        expected = torch.nn.functional.conv_transpose2d(x, w, stride=2, padding=1, output_padding=1)
        result = conv_nd(x, w, dim=(-2, -1), stride=2, padding=1, output_padding=1, transposed=True)
        torch.testing.assert_close(result, expected)

    def test_conv_transpose3d(self) -> None:
        x, w = torch.randn(2, 4, 6, 6, 6), torch.randn(4, 8, 3, 3, 3)
        expected = torch.nn.functional.conv_transpose3d(x, w, stride=2, padding=1)
        result = conv_nd(x, w, dim=(-3, -2, -1), stride=2, padding=1, transposed=True)
        torch.testing.assert_close(result, expected)


class TestRecursiveForward:
    def test_conv3d_via_conv2d(self, disable_conv3d: Iterator[None]) -> None:  # noqa: ARG002
        x, w = torch.randn(2, 4, 8, 8, 8), torch.randn(8, 4, 3, 3, 3)
        expected = torch.nn.functional.conv3d(x, w, stride=2, padding=1)
        result = conv_nd(x, w, dim=(-3, -2, -1), stride=2, padding=1)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)

    def test_conv3d_via_conv1d(self, disable_conv2d_3d: Iterator[None]) -> None:  # noqa: ARG002
        x, w = torch.randn(2, 4, 6, 6, 6), torch.randn(8, 4, 3, 3, 3)
        expected = torch.nn.functional.conv3d(x, w, padding=1)
        result = conv_nd(x, w, dim=(-3, -2, -1), padding=1)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)

    def test_asymmetric_params(self, disable_conv3d: Iterator[None]) -> None:  # noqa: ARG002
        x, w = torch.randn(2, 4, 12, 14, 16), torch.randn(8, 4, 3, 5, 3)
        stride, padding, dilation = (2, 1, 2), (1, 2, 1), (1, 2, 1)
        expected = torch.nn.functional.conv3d(x, w, stride=stride, padding=padding, dilation=dilation)
        result = conv_nd(x, w, dim=(-3, -2, -1), stride=stride, padding=padding, dilation=dilation)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)


class TestRecursiveTranspose:
    def test_transpose3d_via_conv2d(self, disable_conv3d: Iterator[None]) -> None:  # noqa: ARG002
        x, w = torch.randn(2, 4, 6, 6, 6), torch.randn(4, 8, 3, 3, 3)
        expected = torch.nn.functional.conv_transpose3d(x, w, stride=2, padding=1)
        result = conv_nd(x, w, dim=(-3, -2, -1), stride=2, padding=1, transposed=True)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)

    def test_transpose3d_with_output_padding(self, disable_conv3d: Iterator[None]) -> None:  # noqa: ARG002
        x, w = torch.randn(2, 4, 6, 6, 6), torch.randn(4, 8, 3, 3, 3)
        expected = torch.nn.functional.conv_transpose3d(x, w, stride=2, padding=1, output_padding=1)
        result = conv_nd(x, w, dim=(-3, -2, -1), stride=2, padding=1, output_padding=1, transposed=True)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)

    def test_transpose3d_output_padding_exceeds_padding(self, disable_conv3d: Iterator[None]) -> None:  # noqa: ARG002
        x, w = torch.randn(2, 4, 6, 6, 6), torch.randn(4, 8, 3, 3, 3)
        expected = torch.nn.functional.conv_transpose3d(x, w, stride=2, padding=0, output_padding=1)
        result = conv_nd(x, w, dim=(-3, -2, -1), stride=2, padding=0, output_padding=1, transposed=True)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)

    def test_transpose3d_asymmetric(self, disable_conv3d: Iterator[None]) -> None:  # noqa: ARG002
        x, w = torch.randn(2, 4, 6, 8, 10), torch.randn(4, 8, 3, 5, 3)
        stride, padding = (2, 1, 2), (1, 2, 1)
        expected = torch.nn.functional.conv_transpose3d(x, w, stride=stride, padding=padding)
        result = conv_nd(x, w, dim=(-3, -2, -1), stride=stride, padding=padding, transposed=True)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)


class TestComplex:
    def test_both_complex(self) -> None:
        x = torch.randn(2, 4, 16, 16, dtype=torch.complex64)
        w = torch.randn(8, 4, 3, 3, dtype=torch.complex64)
        result = conv_nd(x, w, dim=(-2, -1), padding=1)
        expected = torch.complex(
            torch.nn.functional.conv2d(x.real, w.real, padding=1) - torch.nn.functional.conv2d(x.imag, w.imag, padding=1),
            torch.nn.functional.conv2d(x.real, w.imag, padding=1) + torch.nn.functional.conv2d(x.imag, w.real, padding=1),
        )
        torch.testing.assert_close(result, expected)

    def test_complex_x_only(self) -> None:
        x = torch.randn(2, 4, 16, 16, dtype=torch.complex64)
        w = torch.randn(8, 4, 3, 3)
        result = conv_nd(x, w, dim=(-2, -1), padding=1)
        expected = torch.complex(
            torch.nn.functional.conv2d(x.real, w, padding=1),
            torch.nn.functional.conv2d(x.imag, w, padding=1),
        )
        torch.testing.assert_close(result, expected)

    def test_complex_w_only(self) -> None:
        x = torch.randn(2, 4, 16, 16)
        w = torch.randn(8, 4, 3, 3, dtype=torch.complex64)
        result = conv_nd(x, w, dim=(-2, -1), padding=1)
        expected = torch.complex(
            torch.nn.functional.conv2d(x, w.real, padding=1),
            torch.nn.functional.conv2d(x, w.imag, padding=1),
        )
        torch.testing.assert_close(result, expected)

    def test_complex_transpose(self) -> None:
        x = torch.randn(2, 4, 8, 8, dtype=torch.complex64)
        w = torch.randn(4, 8, 3, 3, dtype=torch.complex64)
        result = conv_nd(x, w, dim=(-2, -1), padding=1, transposed=True)
        expected = torch.complex(
            torch.nn.functional.conv_transpose2d(x.real, w.real, padding=1)
            - torch.nn.functional.conv_transpose2d(x.imag, w.imag, padding=1),
            torch.nn.functional.conv_transpose2d(x.real, w.imag, padding=1)
            + torch.nn.functional.conv_transpose2d(x.imag, w.real, padding=1),
        )
        torch.testing.assert_close(result, expected)


class TestGroups:
    def test_groups_forward(self) -> None:
        x, w = torch.randn(2, 8, 16, 16), torch.randn(16, 4, 3, 3)
        expected = torch.nn.functional.conv2d(x, w, groups=2, padding=1)
        result = conv_nd(x, w, dim=(-2, -1), groups=2, padding=1)
        torch.testing.assert_close(result, expected)

    def test_groups_transpose(self) -> None:
        x, w = torch.randn(2, 8, 8, 8), torch.randn(8, 4, 3, 3)
        expected = torch.nn.functional.conv_transpose2d(x, w, groups=2, padding=1)
        result = conv_nd(x, w, dim=(-2, -1), groups=2, padding=1, transposed=True)
        torch.testing.assert_close(result, expected)

    def test_groups_recursive(self, disable_conv3d: Iterator[None]) -> None:  # noqa: ARG002
        x, w = torch.randn(2, 8, 6, 6, 6), torch.randn(16, 4, 3, 3, 3)
        expected = torch.nn.functional.conv3d(x, w, groups=2, padding=1)
        result = conv_nd(x, w, dim=(-3, -2, -1), groups=2, padding=1)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)


class TestDimLayouts:
    def test_channel_last(self) -> None:
        x, w = torch.randn(2, 16, 16, 4), torch.randn(8, 4, 3, 3)
        expected = torch.nn.functional.conv2d(x.permute(0, 3, 1, 2), w, padding=1).permute(0, 2, 3, 1)
        result = conv_nd(x, w, dim=(1, 2), channel_dim=-1, padding=1)
        torch.testing.assert_close(result, expected)

    def test_extra_batch_dims(self) -> None:
        x, w = torch.randn(2, 3, 4, 16, 16), torch.randn(8, 4, 3, 3)
        expected = torch.nn.functional.conv2d(x.reshape(6, 4, 16, 16), w, padding=1).reshape(2, 3, 8, 16, 16)
        result = conv_nd(x, w, dim=(-2, -1), channel_dim=2, padding=1)
        torch.testing.assert_close(result, expected)

    def test_non_contiguous_spatial(self) -> None:
        x, w = torch.randn(2, 8, 4, 8), torch.randn(6, 4, 3, 3)
        expected = torch.nn.functional.conv2d(x.permute(0, 2, 1, 3), w, padding=1).permute(0, 2, 1, 3)
        result = conv_nd(x, w, dim=(1, 3), channel_dim=2, padding=1)
        torch.testing.assert_close(result, expected)


class TestAutograd:
    def test_backward_forward(self) -> None:
        x = torch.randn(2, 4, 8, 8, requires_grad=True)
        w = torch.randn(8, 4, 3, 3, requires_grad=True)
        conv_nd(x, w, dim=(-2, -1), padding=1).sum().backward()
        assert x.grad is not None and w.grad is not None

    def test_backward_transpose(self) -> None:
        x = torch.randn(2, 4, 8, 8, requires_grad=True)
        w = torch.randn(4, 8, 3, 3, requires_grad=True)
        conv_nd(x, w, dim=(-2, -1), padding=1, transposed=True).sum().backward()
        assert x.grad is not None and w.grad is not None

    def test_gradcheck_recursive(self, disable_conv3d: Iterator[None]) -> None:  # noqa: ARG002
        x = torch.randn(1, 2, 5, 5, 5, dtype=torch.float64, requires_grad=True)
        w = torch.randn(3, 2, 3, 3, 3, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(lambda a, b: conv_nd(a, b, dim=(-3, -2, -1), padding=1), (x, w))

    def test_gradcheck_transpose_recursive(self, disable_conv3d: Iterator[None]) -> None:  # noqa: ARG002
        x = torch.randn(1, 2, 4, 4, 4, dtype=torch.float64, requires_grad=True)
        w = torch.randn(2, 3, 3, 3, 3, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(lambda a, b: conv_nd(a, b, dim=(-3, -2, -1), padding=1, transposed=True), (x, w))

    def test_gradcheck_complex(self) -> None:
        x = torch.randn(1, 2, 5, 5, dtype=torch.complex128, requires_grad=True)
        w = torch.randn(3, 2, 3, 3, dtype=torch.complex128, requires_grad=True)
        assert torch.autograd.gradcheck(lambda a, b: conv_nd(a, b, dim=(-2, -1), padding=1), (x, w))


class TestEdgeCases:
    def test_kernel_1x1(self) -> None:
        x, w = torch.randn(2, 4, 8, 8), torch.randn(8, 4, 1, 1)
        torch.testing.assert_close(conv_nd(x, w, dim=(-2, -1)), torch.nn.functional.conv2d(x, w))

    def test_stride_larger_than_kernel(self) -> None:
        x, w = torch.randn(2, 4, 16, 16), torch.randn(8, 4, 3, 3)
        expected = torch.nn.functional.conv2d(x, w, stride=5, padding=1)
        result = conv_nd(x, w, dim=(-2, -1), stride=5, padding=1)
        torch.testing.assert_close(result, expected)

    def test_no_batch_dim(self) -> None:
        x, w = torch.randn(4, 8, 8), torch.randn(8, 4, 3, 3)
        expected = torch.nn.functional.conv2d(x.unsqueeze(0), w, padding=1).squeeze(0)
        result = conv_nd(x, w, dim=(-2, -1), channel_dim=0, padding=1)
        torch.testing.assert_close(result, expected)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
class TestCompile:
    def test_compile_forward(self) -> None:
        fn = torch.compile(conv_nd)
        x, w = torch.randn(2, 4, 16, 16), torch.randn(8, 4, 3, 3)
        torch.testing.assert_close(fn(x, w, dim=(-2, -1), padding=1), conv_nd(x, w, dim=(-2, -1), padding=1))

    def test_compile_transpose(self) -> None:
        fn = torch.compile(conv_nd)
        x, w = torch.randn(2, 4, 8, 8), torch.randn(4, 8, 3, 3)
        torch.testing.assert_close(
            fn(x, w, dim=(-2, -1), padding=1, transposed=True),
            conv_nd(x, w, dim=(-2, -1), padding=1, transposed=True),
        )


class TestPadNd:
    @pytest.mark.parametrize(
        "shape, pad, dims, mode, expected_shape",
        [
            ((2, 3), (1, 1), None, "constant", (2, 5)),
            ((2, 3), (1, 1), (0,), "constant", (4, 3)),
            ((2, 3, 4), (1, 1), (0,), "reflect", (4, 3, 4)),
            ((2, 3, 4), (1, 1, 2, 2), (0, 2), "constant", (4, 3, 8)),
            ((2, 3, 4), (1, 1), (-3,), "circular", (4, 3, 4)),
            ((1, 1, 1, 1, 1), (1, 1, 1, 1), (0, 4), "constant", (3, 1, 1, 1, 3)),
        ],
    )
    def test_shapes_and_modes(
        self,
        shape: tuple[int, ...],
        pad: tuple[int, ...],
        dims: tuple[int, ...] | None,
        mode: str,
        expected_shape: tuple[int, ...],
    ) -> None:
        """Verify output shapes for various modes and dimension configurations."""
        x = torch.randn(shape)
        out = pad_nd(x, pad, dims=dims, mode=mode)
        assert out.shape == expected_shape

    @pytest.mark.parametrize(
        "pad, expected_slice",
        [
            ((-1, 0), slice(1, 5)),
            ((0, -1), slice(0, 4)),
            ((-1, -1), slice(1, 4)),
            ((-2, -2), slice(2, 3)),
        ],
    )
    def test_cropping(self, pad: tuple[int, ...], expected_slice: slice) -> None:
        """Verify negative padding correctly crops tensor content."""
        x = torch.arange(5).float()
        out = pad_nd(x, pad, dims=(0,))
        torch.testing.assert_close(out, x[expected_slice])

    def test_mixed_pad_and_crop(self) -> None:
        """Verify simultaneous padding and cropping on same dimension."""
        x = torch.arange(5).float()
        out = pad_nd(x, (1, -1), dims=(0,), mode="constant", value=0)
        assert out.shape == (5,)
        assert out[0] == 0
        torch.testing.assert_close(out[1:], x[:4])

    @pytest.mark.parametrize(
        "mode, expected_first, expected_last",
        [
            ("reflect", 1.0, 1.0),
            ("replicate", 0.0, 2.0),
            ("circular", 2.0, 0.0),
        ],
    )
    def test_mode_values(self, mode: str, expected_first: float, expected_last: float) -> None:
        """Verify boundary values for non-constant padding modes."""
        x = torch.tensor([0.0, 1.0, 2.0])
        out = pad_nd(x, (1, 1), dims=(0,), mode=mode)
        assert out[0] == expected_first
        assert out[-1] == expected_last

    def test_autograd(self) -> None:
        """Verify gradient flow through padding and cropping."""
        x = torch.randn(1, 5, requires_grad=True)
        out = pad_nd(x, (2, -1), dims=(1,), mode="constant")
        out.sum().backward()

        assert x.grad is not None
        assert x.grad[0, 4] == 0.0
        torch.testing.assert_close(x.grad[0, :4], torch.ones(4))

    @pytest.mark.parametrize(
        "kwargs, error_match",
        [
            ({"pad": (1, 1, 1)}, "even length"),
            ({"pad": (1, 1), "dims": (0, 1)}, "len"),
            ({"pad": (1, 1, 1, 1), "dims": (0, 0)}, "duplicate"),
        ],
    )
    def test_validation_errors(self, kwargs: dict, error_match: str) -> None:
        """Verify invalid inputs raise informative errors."""
        with pytest.raises(ValueError, match=error_match):
            pad_nd(torch.zeros(2, 2), **kwargs)

    def test_multi_dim_non_constant_chunking(self) -> None:
        """Verify correct handling of >3 dims requiring multiple pad calls."""
        x = torch.randn(2, 3, 4, 5, 6)
        out = pad_nd(x, (1, 1, 1, 1, 1, 1, 1, 1), dims=(0, 1, 2, 3), mode="reflect")
        assert out.shape == (4, 5, 6, 7, 6)
        torch.testing.assert_close(out[1:-1, 1:-1, 1:-1, 1:-1, :], x)


class TestPadOrCropToSize:
    @pytest.mark.parametrize(
        "input_shape, target_size, dims, expected_shape",
        [
            ((10, 10), (12, 12), None, (12, 12)),
            ((10, 10), (8, 8), None, (8, 8)),
            ((10, 10), (12, 8), None, (12, 8)),
            ((10, 10, 10), (5,), (1,), (10, 5, 10)),
            ((5,), (10,), (0,), (10,)),
        ],
    )
    def test_resizing(
        self,
        input_shape: tuple[int, ...],
        target_size: tuple[int, ...],
        dims: tuple[int, ...] | None,
        expected_shape: tuple[int, ...],
    ) -> None:
        """Verify tensor resized to exact target shape."""
        out = pad_or_crop_to_size(torch.zeros(input_shape), target_size, dims=dims)
        assert out.shape == expected_shape

    def test_centering(self) -> None:
        """Verify resizing maintains tensor center."""
        x = torch.zeros(4, 4)
        x[1:3, 1:3] = 1.0

        cropped = pad_or_crop_to_size(x, (2, 2))
        torch.testing.assert_close(cropped, torch.ones(2, 2))

        padded = pad_or_crop_to_size(x, (6, 6))
        torch.testing.assert_close(padded[2:4, 2:4], torch.ones(2, 2))
        assert padded[0, 0] == 0.0

    def test_validation_error(self) -> None:
        """Verify mismatched dims and size raises error."""
        with pytest.raises(ValueError, match="len"):
            pad_or_crop_to_size(torch.zeros(2, 2), (1, 2, 3), dims=(0, 1))


class TestCompilation:
    @pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
    def test_compile_fullgraph(self) -> None:
        """Verify compatibility with torch.compile fullgraph mode."""
        x = torch.randn(10, 10)

        @torch.compile(fullgraph=True)
        def fn(t: torch.Tensor) -> torch.Tensor:
            return pad_nd(t, (1, 1), dims=(0,), mode="reflect")

        torch.testing.assert_close(fn(x), pad_nd(x, (1, 1), dims=(0,), mode="reflect"))

    def test_jit_trace(self) -> None:
        """Verify compatibility with torch.jit.trace."""
        x = torch.randn(2, 3, 4)
        traced = torch.jit.trace(lambda t: pad_nd(t, (1, 1, 2, 2), mode="constant"), (x,))
        torch.testing.assert_close(traced(x), pad_nd(x, (1, 1, 2, 2), mode="constant"))


class TestAdjointPadNd:
    """Tests for adjoint_pad_nd function."""

    @pytest.mark.parametrize(
        "shape, pad, dims, mode",
        [
            ((4, 8, 8), (2, 2, 2, 2), (-2, -1), "constant"),
            ((4, 8, 8), (2, 2, 2, 2), (-2, -1), "reflect"),
            ((4, 8, 8), (2, 2, 2, 2), (-2, -1), "replicate"),
            ((4, 8, 8), (2, 2, 2, 2), (-2, -1), "circular"),
            ((2, 3, 6, 6), (1, 1), (-1,), "reflect"),
            ((2, 3, 6, 6, 6), (1, 1, 2, 2), (-2, -1), "circular"),
        ],
    )
    def test_inner_product_identity(self, shape: tuple[int, ...], pad: tuple[int, ...], dims: tuple[int, ...], mode: str) -> None:
        """Verify <Ax, y> == <x, A^H y> for padding operator."""
        x = torch.randn(shape)
        padded = pad_nd(x, pad, mode=mode, dims=dims)
        y = torch.randn_like(padded)

        n_dims = len(pad) // 2
        target_dims = list(range(len(shape) - 1, len(shape) - n_dims - 1, -1)) if dims is None else [d % len(shape) for d in dims]
        original_sizes = [shape[d] for d in target_dims]

        adjoint_y = adjoint_pad_nd(y, pad, mode=mode, dims=dims, original_sizes=original_sizes)

        lhs = torch.vdot(padded.flatten(), y.flatten())
        rhs = torch.vdot(x.flatten(), adjoint_y.flatten())

        torch.testing.assert_close(lhs, rhs, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("mode", ["constant", "reflect", "replicate", "circular"])
    def test_output_shape(self, mode: str) -> None:
        """Adjoint returns tensor with original (unpadded) shape."""
        original_shape = (2, 4, 10, 10)
        pad = (2, 2, 3, 3)
        dims = (-2, -1)

        x = torch.randn(original_shape)
        padded = pad_nd(x, pad, mode=mode, dims=dims)

        original_sizes = [original_shape[-2], original_shape[-1]]
        result = adjoint_pad_nd(padded, pad, mode=mode, dims=dims, original_sizes=original_sizes)

        assert result.shape == original_shape

    def test_autograd_differentiable(self) -> None:
        """Adjoint operation is differentiable."""
        y = torch.randn(2, 4, 14, 14, requires_grad=True)

        result = adjoint_pad_nd(y, (2, 2, 2, 2), mode="reflect", dims=(-2, -1), original_sizes=[10, 10])
        result.sum().backward()

        assert y.grad is not None

    def test_gradcheck(self) -> None:
        """Numerical gradient check passes for adjoint_pad_nd."""
        pad = (1, 1, 1, 1)
        dims = (-2, -1)

        def fn(y: torch.Tensor) -> torch.Tensor:
            return adjoint_pad_nd(y, pad, mode="reflect", dims=dims, original_sizes=[6, 6])

        y = torch.randn(1, 2, 8, 8, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(fn, y)

    def test_zeros_mode_is_crop(self) -> None:
        """For constant/zeros mode, adjoint is simple cropping."""
        x = torch.randn(2, 4, 10, 10)
        padded = pad_nd(x, (2, 2, 2, 2), mode="constant", value=0.0, dims=(-2, -1))
        result = adjoint_pad_nd(padded, (2, 2, 2, 2), mode="constant", dims=(-2, -1), original_sizes=[10, 10])
        torch.testing.assert_close(result, x)

    def test_infers_original_sizes(self) -> None:
        """Original sizes can be inferred from pad and input shape."""
        padded = torch.randn(2, 4, 14, 14)  # 10 + 2 + 2 = 14
        pad = (2, 2, 2, 2)
        dims = (-2, -1)

        result = adjoint_pad_nd(padded, pad, mode="reflect", dims=dims)

        assert result.shape == (2, 4, 10, 10)
