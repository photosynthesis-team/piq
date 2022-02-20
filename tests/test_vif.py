import torch
import pytest
from typing import Tuple

from piq import VIFLoss, vif_p
from skimage.io import imread


@pytest.fixture(scope='module')
def x() -> torch.Tensor:
    return torch.rand(4, 3, 256, 256)


@pytest.fixture(scope='module')
def y() -> torch.Tensor:
    return torch.rand(4, 3, 256, 256)


@pytest.fixture(scope='module')
def x_1d() -> torch.Tensor:
    return torch.rand(4, 1, 256, 256)


@pytest.fixture(scope='module')
def y_1d() -> torch.Tensor:
    return torch.rand(4, 1, 256, 256)


# ================== Test function: `vif_p` ==================
def test_vif_p(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    x, y = input_tensors
    vif_p(x.to(device), y.to(device), data_range=1.)


def test_vif_p_one_for_equal_tensors(x) -> None:
    y = x.clone()
    measure = vif_p(x, y)
    assert torch.isclose(measure, torch.tensor(1.0)), f'VIF for equal tensors should be 1.0, got {measure}.'


def test_vif_p_works_for_zeros_tensors() -> None:
    x = torch.zeros(4, 3, 256, 256)
    y = torch.zeros(4, 3, 256, 256)
    measure = vif_p(x, y, data_range=1.)
    assert torch.isclose(measure, torch.tensor(1.0)), f'VIF for 2 zero tensors should be 1.0, got {measure}.'


def test_vif_p_fails_for_small_images() -> None:
    x = torch.rand(2, 3, 32, 32)
    y = torch.rand(2, 3, 32, 32)
    with pytest.raises(ValueError):
        vif_p(x, y)


@pytest.mark.parametrize(
    "data_range", [128, 255],
)
def test_vif_supports_different_data_ranges(x, y, data_range, device: str) -> None:
    x_scaled = (x * data_range).type(torch.uint8)
    y_scaled = (y * data_range).type(torch.uint8)
    measure_scaled = vif_p(x_scaled.to(device), y_scaled.to(device), data_range=data_range)
    measure = vif_p(
        x_scaled.to(device) / float(data_range),
        y_scaled.to(device) / float(data_range),
        data_range=1.0
    )
    diff = torch.abs(measure_scaled - measure)
    assert diff <= 1e-5, f'Result for same tensor with different data_range should be the same, got {diff}'


def test_vif_fails_for_incorrect_data_range(x, y, device: str) -> None:
    # Scale to [0, 255]
    x_scaled = (x * 255).type(torch.uint8)
    y_scaled = (y * 255).type(torch.uint8)
    with pytest.raises(AssertionError):
        vif_p(x_scaled.to(device), y_scaled.to(device), data_range=1.0)


def test_vif_simmular_to_matlab_implementation():
    # Greyscale images
    goldhill = torch.tensor(imread('tests/assets/goldhill.gif'))[None, None, ...]
    goldhill_jpeg = torch.tensor(imread('tests/assets/goldhill_jpeg.gif'))[None, None, ...]

    score = vif_p(goldhill_jpeg, goldhill, data_range=255, reduction='none')
    score_baseline = torch.tensor(0.2665)

    assert torch.isclose(score, score_baseline, atol=1e-4), \
        f'Expected PyTorch score to be equal to MATLAB prediction. Got {score} and {score_baseline}'

    # RGB images
    I01 = torch.tensor(imread('tests/assets/I01.BMP')).permute(2, 0, 1)[None, ...]
    i1_01_5 = torch.tensor(imread('tests/assets/i01_01_5.bmp')).permute(2, 0, 1)[None, ...]

    score = vif_p(i1_01_5, I01, data_range=255, reduction='none')

    # RGB images are not supported by MATLAB code. Here is result for luminance channel taken from YIQ colour space
    score_baseline = torch.tensor(0.3147)

    assert torch.isclose(score, score_baseline, atol=1e-4), \
        f'Expected PyTorch score to be equal to MATLAB prediction. Got {score} and {score_baseline}'


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64],
)
def test_vif_preserves_dtype(x, y, dtype, device: str) -> None:
    output = vif_p(x.to(device=device, dtype=dtype), y.to(device=device, dtype=dtype))
    assert output.dtype == dtype


# ================== Test class: `VIFLoss` ==================
def test_vif_loss_forward(x, y, device: str) -> None:
    loss = VIFLoss()
    loss(x.to(device), y.to(device))


def test_vif_loss_zero_for_equal_tensors(x):
    loss = VIFLoss()
    y = x.clone()
    measure = loss(x, y)
    assert torch.isclose(measure, torch.tensor(0.), atol=1e-6), f'VIF for equal tensors must be 0, got {measure}'


def test_vif_loss_reduction(x, y) -> None:
    loss = VIFLoss(reduction='mean')
    measure = loss(x, y)
    assert measure.dim() == 0, f'VIF with `mean` reduction must return 1 number, got {len(measure)}'

    loss = VIFLoss(reduction='sum')
    measure = loss(x, y)
    assert measure.dim() == 0, f'VIF with `mean` reduction must return 1 number, got {len(measure)}'

    loss = VIFLoss(reduction='none')
    measure = loss(x, y)
    assert len(measure) == x.size(0), \
        f'VIF with `none` reduction must have length equal to number of images, got {len(measure)}'

    loss = VIFLoss(reduction='random string')
    with pytest.raises(ValueError):
        loss(x, y)


NONE_GRAD_ERR_MSG = 'Expected non None gradient of leaf variable'


def test_vif_loss_computes_grad(x, y, device: str) -> None:
    x.requires_grad_()
    loss_value = VIFLoss()(x.to(device), y.to(device))
    loss_value.backward()
    assert x.grad is not None, NONE_GRAD_ERR_MSG


def test_vif_loss_computes_grad_for_zeros_tensors() -> None:
    x = torch.zeros(4, 3, 256, 256, requires_grad=True)
    y = torch.zeros(4, 3, 256, 256)
    loss_value = VIFLoss()(x, y)
    loss_value.backward()
    assert x.grad is not None, NONE_GRAD_ERR_MSG
