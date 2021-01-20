import torch
import pytest
from typing import Tuple

from piq import VIFLoss, vif_p
from skimage.io import imread


@pytest.fixture(scope='module')
def prediction() -> torch.Tensor:
    return torch.rand(4, 3, 256, 256)


@pytest.fixture(scope='module')
def target() -> torch.Tensor:
    return torch.rand(4, 3, 256, 256)


@pytest.fixture(scope='module')
def prediction_1d() -> torch.Tensor:
    return torch.rand(4, 1, 256, 256)


@pytest.fixture(scope='module')
def target_1d() -> torch.Tensor:
    return torch.rand(4, 1, 256, 256)


# ================== Test function: `vif_p` ==================
def test_vif_p(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction, target = input_tensors
    vif_p(prediction.to(device), target.to(device), data_range=1.)


def test_vif_p_one_for_equal_tensors(prediction: torch.Tensor) -> None:
    target = prediction.clone()
    measure = vif_p(prediction, target)
    assert torch.isclose(measure, torch.tensor(1.0)), f'VIF for equal tensors shouls be 1.0, got {measure}.'


def test_vif_p_works_for_zeros_tensors() -> None:
    prediction = torch.zeros(4, 3, 256, 256)
    target = torch.zeros(4, 3, 256, 256)
    measure = vif_p(prediction, target, data_range=1.)
    assert torch.isclose(measure, torch.tensor(1.0)), f'VIF for 2 zero tensors shouls be 1.0, got {measure}.'


def test_vif_p_fails_for_small_images() -> None:
    prediction = torch.rand(2, 3, 32, 32)
    target = torch.rand(2, 3, 32, 32)
    with pytest.raises(ValueError):
        vif_p(prediction, target)


@pytest.mark.parametrize(
    "data_range", [128, 255],
)
def test_vif_supports_different_data_ranges(
        prediction: torch.Tensor, target: torch.Tensor, data_range, device: str) -> None:
    prediction_scaled = (prediction * data_range).type(torch.uint8)
    target_scaled = (target * data_range).type(torch.uint8)
    measure_scaled = vif_p(prediction_scaled.to(device), target_scaled.to(device), data_range=data_range)
    measure = vif_p(
        prediction_scaled.to(device) / float(data_range),
        target_scaled.to(device) / float(data_range),
        data_range=1.0
    )
    diff = torch.abs(measure_scaled - measure)
    assert diff <= 1e-5, f'Result for same tensor with different data_range should be the same, got {diff}'


def test_vif_fails_for_incorrect_data_range(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    # Scale to [0, 255]
    prediction_scaled = (prediction * 255).type(torch.uint8)
    target_scaled = (target * 255).type(torch.uint8)
    with pytest.raises(AssertionError):
        vif_p(prediction_scaled.to(device), target_scaled.to(device), data_range=1.0)


def test_vif_simmular_to_matlab_implementation():
    # Greyscale images
    goldhill = torch.tensor(imread('tests/assets/goldhill.gif'))
    goldhill_jpeg = torch.tensor(imread('tests/assets/goldhill_jpeg.gif'))

    score = vif_p(goldhill_jpeg, goldhill, data_range=255, reduction='none')
    score_baseline = torch.tensor(0.2665)

    assert torch.isclose(score, score_baseline, atol=1e-4), \
        f'Expected PyTorch score to be equal to MATLAB prediction. Got {score} and {score_baseline}'

    # RGB images
    I01 = torch.tensor(imread('tests/assets/I01.BMP')).permute(2, 0, 1)
    i1_01_5 = torch.tensor(imread('tests/assets/i01_01_5.bmp')).permute(2, 0, 1)

    score = vif_p(i1_01_5, I01, data_range=255, reduction='none')

    # RGB images are not supported by MATLAB code. Here is result for luminance channel taken from YIQ colour space
    score_baseline = torch.tensor(0.3147)

    assert torch.isclose(score, score_baseline, atol=1e-4), \
        f'Expected PyTorch score to be equal to MATLAB prediction. Got {score} and {score_baseline}'


# ================== Test class: `VIFLoss` ==================
def test_vif_loss_forward(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    loss = VIFLoss()
    loss(prediction.to(device), target.to(device))

    
def test_vif_loss_zero_for_equal_tensors(prediction: torch.Tensor):
    loss = VIFLoss()
    target = prediction.clone()
    measure = loss(prediction, target)
    assert torch.isclose(measure, torch.tensor(0.), atol=1e-6), f'VIF for equal tensors must be 0, got {measure}'


def test_vif_loss_reduction(prediction: torch.Tensor, target: torch.Tensor) -> None:
    loss = VIFLoss(reduction='mean')
    measure = loss(prediction, target)
    assert measure.dim() == 0, f'VIF with `mean` reduction must return 1 number, got {len(measure)}'

    loss = VIFLoss(reduction='sum')
    measure = loss(prediction, target)
    assert measure.dim() == 0, f'VIF with `mean` reduction must return 1 number, got {len(measure)}'

    loss = VIFLoss(reduction='none')
    measure = loss(prediction, target)
    assert len(measure) == prediction.size(0), \
        f'VIF with `none` reduction must have length equal to number of images, got {len(measure)}'
    
    loss = VIFLoss(reduction='random string')
    with pytest.raises(KeyError):
        loss(prediction, target)


NONE_GRAD_ERR_MSG = 'Expected non None gradient of leaf variable'


def test_vif_loss_computes_grad(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    prediction.requires_grad_()
    loss_value = VIFLoss()(prediction.to(device), target.to(device))
    loss_value.backward()
    assert prediction.grad is not None, NONE_GRAD_ERR_MSG


def test_vif_loss_computes_grad_for_zeros_tensors() -> None:
    prediction = torch.zeros(4, 3, 256, 256, requires_grad=True)
    target = torch.zeros(4, 3, 256, 256)
    loss_value = VIFLoss()(prediction, target)
    loss_value.backward()
    assert prediction.grad is not None, NONE_GRAD_ERR_MSG
