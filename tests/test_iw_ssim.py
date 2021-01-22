import torch
import pytest
from piq import information_weighted_ssim, InformationWeightedSSIMLoss
from skimage.io import imread
from typing import Tuple


# ================== Test function: `iw_ssim` ==================
def test_iw_ssim_to_be_one_for_identical_inputs(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    x, _ = input_tensors
    index = information_weighted_ssim(x.to(device), y.to(device), data_range=1., reduction='none')
    index_255 = information_weighted_ssim(
        x=(x * 255).to(device=device, dtype=torch.uint8),
        y=(y * 255).to(device=device, dtype=torch.uint8),
        data_range=255,
        reduction='none'
    )
    assert torch.allclose(index, torch.ones_like(index, device=device), atol=1e-5), \
        f'Expected index to be equal 1, got {index}'
    assert torch.allclose(index_255, torch.ones_like(index_255, device=device), atol=1e-5), \
        f'Expected index to be equal 1, got {index_255}'


def test_iw_ssim_symmetry(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    x, y = input_tensors
    result = information_weighted_ssim(x.to(device), y.to(device), data_range=1., reduction='none')
    result_sym = information_weighted_ssim(y.to(device), x.to(device), data_range=1., reduction='none')
    assert torch.allclose(result_sym, result), f'Expected the same results, got {result} and {result_sym}'


def test_iw_ssim_zeros_ones_inputs(device: str) -> None:
    zeros = torch.zeros(1, 3, 128, 128, device=device)
    ones = torch.ones(1, 3, 128, 128, device=device)
    iw_ssim_ones = information_weighted_ssim(ones, ones, data_range=1.)
    assert torch.isfinite(iw_ssim_ones).all(), f'Expected finite value for ones tensos, got {iw_ssim_ones}'
    iw_ssim_zeros_ones = information_weighted_ssim(zeros, ones, data_range=1.)
    assert torch.isfinite(iw_ssim_zeros_ones).all(), \
        f'Expected finite value for zeros and ones tensos, got {iw_ssim_zeros_ones}'


def test_iw_ssim_small_input(device: str) -> None:
    x = torch.rand(1, 3, 10, 10, device=device)
    y = torch.rand(1, 3, 10, 10, device=device)
    with pytest.raises(ValueError):
        information_weighted_ssim(x, y, data_range=1.)

-------
@pytest.mark.parametrize(
    "data_range", [128, 255],
)
def test_iw_ssim_supports_different_data_ranges(
        input_tensors: Tuple[torch.Tensor, torch.Tensor], data_rangeL int, device: str) -> None:
    x, y = input_tensors
    x_scaled = (x * data_range).type(torch.uint8)
    y_scaled = (y * data_range).type(torch.uint8)

    measure_scaled = haarpsi(prediction_scaled.to(device), target_scaled.to(device), data_range=data_range)
    measure = haarpsi(
        prediction_scaled.to(device) / float(data_range),
        target_scaled.to(device) / float(data_range),
        data_range=1.0
    )
    diff = torch.abs(measure_scaled - measure)
    assert diff <= 1e-6, f'Result for same tensor with different data_range should be the same, got {diff}'


def test_haarpsi_fails_for_incorrect_data_range(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    # Scale to [0, 255]
    prediction_scaled = (prediction * 255).type(torch.uint8)
    target_scaled = (target * 255).type(torch.uint8)
    with pytest.raises(AssertionError):
        haarpsi(prediction_scaled.to(device), target_scaled.to(device), data_range=1.0)


def test_haarpsi_compare_with_matlab(device: str) -> None:
    prediction = torch.tensor(imread('tests/assets/I01.BMP')).permute(2, 0, 1)
    target = torch.tensor(imread('tests/assets/i01_01_5.bmp')).permute(2, 0, 1)
    predicted_score = haarpsi(prediction.to(device), target.to(device), data_range=255, reduction='none')
    target_score = torch.tensor([0.71706527]).to(predicted_score)
    assert torch.isclose(predicted_score, target_score, atol=1e-4),\
        f'Expected result similar to MATLAB, got diff{predicted_score - target_score}'


# ================== Test class: `HaarPSILoss` =================
def test_haarpsi_loss(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction, target = input_tensors
    prediction.requires_grad_()
    loss = HaarPSILoss(data_range=1.)(prediction.to(device), target.to(device))
    loss.backward()
    assert torch.isfinite(prediction.grad).all(), \
        f'Expected finite gradient values after back propagation, got {prediction.grad}'


def test_haarpsi_loss_zero_for_equal_input(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction, _ = input_tensors
    target = prediction.clone()
    prediction.requires_grad_()
    loss = HaarPSILoss(data_range=1.)(prediction.to(device), target.to(device))
    assert torch.isclose(loss, torch.zeros_like(loss), atol=1e-5), \
        f'Expected loss equals zero for identical inputs, got {loss}'
