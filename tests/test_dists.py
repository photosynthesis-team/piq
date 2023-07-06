import torch
import pytest
from typing import Any
from contextlib import contextmanager

from skimage.io import imread
from piq import DISTS


@contextmanager
def raise_nothing():
    yield


# ================== Test class: `DISTS` ==================
def test_dists_loss_forward(x, y, device: str) -> None:
    loss = DISTS()
    loss(x.to(device), y.to(device))


def test_dists_computes_grad(x, y, device: str) -> None:
    x.requires_grad_()
    loss_value = DISTS(enable_grad=True)(x.to(device), y.to(device))
    loss_value.backward()
    assert x.grad is not None, 'Expected non None gradient of leaf variable'


def test_dists_loss_does_not_compute_grad(x, y, device: str) -> None:
    x.requires_grad_()
    loss_value = DISTS(enable_grad=False)(x.to(device), y.to(device))
    with pytest.raises(RuntimeError):
        loss_value.backward()


@pytest.mark.parametrize(
    "x, y, expectation, value",
    [
        (torch.zeros(2, 3, 96, 96), torch.zeros(2, 3, 96, 96), raise_nothing(), 0.0),
        (torch.ones(2, 3, 96, 96), torch.ones(2, 3, 96, 96), raise_nothing(), 0.0),
    ],
)
def test_dists_loss_forward_for_special_cases(x, y, expectation: Any, value: float) -> None:
    loss = DISTS()
    with expectation:
        loss_value = loss(x, y)
        assert torch.isclose(loss_value, torch.tensor(value), atol=1e-6), \
            f'Expected loss value to be equal to target value. Got {loss_value} and {value}'


def test_dists_simmilar_to_official_implementation() -> None:
    # Baseline scores from: https://github.com/dingkeyan93/DISTS
    loss = DISTS()

    # Greyscale images
    goldhill = torch.tensor(imread('tests/assets/goldhill.gif'))[None, None, ...] / 255.0
    goldhill_jpeg = torch.tensor(imread('tests/assets/goldhill_jpeg.gif'))[None, None, ...] / 255.0

    loss_value = loss(goldhill_jpeg, goldhill)
    baseline_value = torch.tensor(0.19509)
    assert torch.isclose(loss_value, baseline_value, atol=1e-3), \
        f'Expected PIQ loss to be equal to original. Got {loss_value} and {baseline_value}'

    # RGB images
    I01 = torch.tensor(imread('tests/assets/I01.BMP')).permute(2, 0, 1)[None, ...] / 255.0
    i1_01_5 = torch.tensor(imread('tests/assets/i01_01_5.bmp')).permute(2, 0, 1)[None, ...] / 255.0

    loss_value = loss(i1_01_5, I01)
    baseline_value = torch.tensor(0.17321)

    assert torch.isclose(loss_value, baseline_value, atol=1e-3), \
        f'Expected PIQ loss to be equal to original. Got {loss_value} and {baseline_value}'
