import torch
import pytest
from typing import Any
from contextlib import contextmanager

from skimage.io import imread
from piq import PieAPP


@contextmanager
def raise_nothing():
    yield


def test_pieapp_loss_forward(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    loss = PieAPP()
    loss(prediction.to(device), target.to(device))


def test_pieapp_computes_grad(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    prediction.requires_grad_()
    loss_value = PieAPP()(prediction.to(device), target.to(device))
    loss_value.backward()
    assert prediction.grad is not None, 'Expected non None gradient of leaf variable'


@pytest.mark.parametrize(
    "prediction,target,expectation,value",
    [
        (torch.zeros(2, 3, 96, 96), torch.zeros(2, 3, 96, 96), raise_nothing(), 0.0),
        (torch.ones(2, 3, 96, 96), torch.ones(2, 3, 96, 96), raise_nothing(), 0.0),
    ],
)
def test_pieapp_loss_forward_for_special_cases(
        prediction: torch.Tensor, target: torch.Tensor, expectation: Any, value: float) -> None:
    loss = PieAPP()
    with expectation:
        loss_value = loss(prediction, target)
        assert torch.isclose(loss_value, torch.tensor(value), atol=1e-6), \
            f'Expected loss value to be equal to target value. Got {loss_value} and {value}'


def test_pieapp_simmilar_to_official_implementation() -> None:
    loss = PieAPP(data_range=255, stride=27)

    # RGB images
    I01 = torch.tensor(imread('tests/assets/I01.BMP')).permute(2, 0, 1)
    i1_01_5 = torch.tensor(imread('tests/assets/i01_01_5.bmp')).permute(2, 0, 1)

    loss_value = loss(i1_01_5, I01)
    # Baseline score from: https://github.com/prashnani/PerceptualImageError
    baseline_value = torch.tensor(2.2393305)

    assert torch.isclose(loss_value, baseline_value, atol=1e-5), \
        f'Expected PIQ loss to be equal to original. Got {loss_value} and {baseline_value}'
