"""Convolution modules can share an abstract torch module base."""

from abc import ABC, abstractmethod

import pytest
import torch

from torchnd import ConvNd, ConvTransposeNd
from torchnd.modules import _ConvNdBase


class AbstractModule(ABC, torch.nn.Module):
    """An abstract interface that places ABC before Module, as mr2 does."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the module."""
        raise NotImplementedError


@pytest.mark.parametrize("conv_type", [ConvNd, ConvTransposeNd])
@pytest.mark.parametrize("abstract_first", [False, True])
def test_multiple_inheritance(conv_type: type[ConvNd] | type[ConvTransposeNd], abstract_first: bool) -> None:
    """Either base order preserves construction, parameters, gradients, and the adjoint."""
    bases = (AbstractModule, conv_type) if abstract_first else (conv_type, AbstractModule)
    combined_type = type("CombinedConvolution", bases, {"forward": conv_type.forward})
    module = combined_type(2, 3, 3, dim=(-1,), padding=1, bias=False, dtype=torch.float64)
    assert isinstance(module, AbstractModule)
    assert isinstance(module, conv_type)
    assert dict(module.named_parameters())["weight"] is module.weight
    torch.testing.assert_close(module.state_dict()["weight"], module.weight)

    generator = torch.Generator().manual_seed(0)
    x = torch.randn(1, 2, 5, generator=generator, dtype=torch.float64, requires_grad=True)
    y = module(x)
    assert y.shape == (1, 3, 5)
    v = torch.randn(y.shape, generator=generator, dtype=y.dtype)
    expected_adjoint = module.adjoint(v)
    (gradient,) = torch.autograd.grad(y, x, grad_outputs=v, retain_graph=True)
    torch.testing.assert_close(gradient, expected_adjoint)
    y.square().sum().backward()
    assert module.weight.grad is not None
    assert module.float() is module
    assert module.weight.dtype == torch.float32


def test_base_remains_abstract() -> None:
    """Using ABCMeta directly still prevents constructing the incomplete base class."""
    with pytest.raises(TypeError, match="abstract"):
        _ConvNdBase(1, 1, 3)  # type: ignore[abstract]
