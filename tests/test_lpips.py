import torch
import pytest
from typing import Any, Tuple, Callable, Union
from contextlib import contextmanager

from skimage.io import imread
from piq import LPIPS
from piq.feature_extractors import InceptionV3


@contextmanager
def raise_nothing():
    yield


# ================== Test class: `LPIPS` ==================
def test_lpips_loss_init() -> None:
    LPIPS()


def test_lpips_loss_forward(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    x, y = input_tensors
    loss = LPIPS()
    loss(x.to(device), y.to(device))


def test_lpips_computes_grad(x, y, device: str) -> None:
    x.requires_grad_()
    loss_value = LPIPS()(x.to(device), y.to(device))
    loss_value.backward()
    assert x.grad is not None, NONE_GRAD_ERR_MSG


def test_lpips_loss_raises_if_wrong_reduction(x, y) -> None:
    for mode in ['mean', 'sum', 'none']:
        LPIPS(reduction=mode)(x, y)

    for mode in [None, 'n', 2]:
        with pytest.raises(ValueError):
            LPIPS(reduction=mode)(x, y)


@pytest.mark.parametrize(
    "x, y, expectation, value",
    [
        (torch.zeros(2, 3, 96, 96), torch.zeros(2, 3, 96, 96), raise_nothing(), 0.0),
        (torch.ones(2, 3, 96, 96), torch.ones(2, 3, 96, 96), raise_nothing(), 0.0),
    ],
)
def test_lpips_loss_forward_for_special_cases(x, y, expectation: Any, value: float) -> None:
    loss = LPIPS()
    with expectation:
        loss_value = loss(x, y)
        assert torch.isclose(loss_value, torch.tensor(value), atol=1e-6), \
            f'Expected loss value to be equal to target value. Got {loss_value} and {value}'

