import torch
import pytest
from typing import Any, Tuple
from contextlib import contextmanager

from skimage.io import imread
from piq import PieAPP


@contextmanager
def raise_nothing():
    yield


def test_pieapp_loss_forward(x, y, device: str) -> None:
    loss = PieAPP()
    loss(x.to(device), y.to(device))


def test_pieapp_computes_grad(x, y, device: str) -> None:
    x.requires_grad_()
    loss_value = PieAPP(enable_grad=True)(x.to(device), y.to(device))
    loss_value.backward()
    assert x.grad is not None, 'Expected non None gradient of leaf variable'


@pytest.mark.parametrize(
    "x, y, expectation,value",
    [
        (torch.zeros(2, 3, 96, 96), torch.zeros(2, 3, 96, 96), raise_nothing(), 0.0),
        (torch.ones(2, 3, 96, 96), torch.ones(2, 3, 96, 96), raise_nothing(), 0.0),
    ],
)
def test_pieapp_loss_forward_for_special_cases(x, y, expectation: Any, value: float) -> None:
    loss = PieAPP()
    with expectation:
        loss_value = loss(x, y)
        assert torch.isclose(loss_value, torch.tensor(value), atol=1e-6), \
            f'Expected loss value to be equal to target value. Got {loss_value} and {value}'


def test_pieapp_simmilar_to_official_implementation() -> None:
    loss = PieAPP(data_range=255, stride=27)

    # RGB images
    I01 = torch.tensor(imread('tests/assets/I01.BMP')).permute(2, 0, 1)[None, ...]
    i1_01_5 = torch.tensor(imread('tests/assets/i01_01_5.bmp')).permute(2, 0, 1)[None, ...]

    loss_value = loss(i1_01_5, I01)
    # Baseline score from: https://github.com/prashnani/PerceptualImageError
    baseline_value = torch.tensor(2.2393305)

    assert torch.isclose(loss_value, baseline_value, atol=1e-5), \
        f'Expected PIQ loss to be equal to original. Got {loss_value} and {baseline_value}'


@pytest.mark.parametrize(
    "data_range", [128, 255],
)
def test_pieapp_supports_different_data_ranges(
        input_tensors: Tuple[torch.Tensor, torch.Tensor], data_range, device: str) -> None:
    x, y = input_tensors
    x_scaled = x * data_range
    y_scaled = y * data_range

    measure_scaled = PieAPP(data_range=data_range, stride=27)(x_scaled.to(device), y_scaled.to(device))
    measure = PieAPP(data_range=1.0, stride=27)(
        x_scaled.to(device) / data_range,
        y_scaled.to(device) / data_range,
    )
    diff = torch.abs(measure_scaled - measure)
    assert diff <= 1e-6, f'Result for same tensor with different data_range should be the same, got {diff}'


def test_pieapp_fails_for_incorrect_data_range(x, y, device: str) -> None:
    # Scale to [0, 255]
    x_scaled = (x * 255).type(torch.uint8)
    y_scaled = (y * 255).type(torch.uint8)
    with pytest.raises(AssertionError):
        PieAPP(data_range=1.0, stride=27)(x_scaled.to(device), y_scaled.to(device))
