from typing import Tuple
import torch
import pytest
from skimage.io import imread

from piq import srsim, SRSIMLoss


# ================== Test function: `srsrim` ==================
def test_srsim_to_be_one_for_identical_inputs(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction, _ = input_tensors
    index = srsim(prediction.to(device), prediction.to(device), data_range=1., reduction='none')

    prediction_255 = (prediction * 255).type(torch.uint8)
    index_255 = srsim(prediction_255, prediction_255, data_range=255, reduction='none')
    assert torch.allclose(index, torch.ones_like(index, device=device)), \
        f'Expected index to be equal 1, got {index}'
    assert torch.allclose(index_255, torch.ones_like(index_255, device=device)), \
        f'Expected index to be equal 1, got {index_255}'


def test_srsim_symmetry(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction, target = input_tensors
    result = srsim(prediction.to(device), target.to(device), data_range=1., reduction='none')
    result_sym = srsim(target.to(device), prediction.to(device), data_range=1., reduction='none')
    assert torch.allclose(result_sym, result), f'Expected the same results, got {result} and {result_sym}'


def test_srsim_zeros_ones_inputs(device: str) -> None:
    zeros = torch.zeros(1, 3, 128, 128, device=device)
    ones = torch.ones(1, 3, 128, 128, device=device)
    srsim_zeros = srsim(zeros, zeros, data_range=1.)
    assert torch.isfinite(srsim_zeros).all(), f'Expected finite value for zeros tensors, got {srsim_zeros}'
    srsim_ones = srsim(ones, ones, data_range=1.)
    assert torch.isfinite(srsim_ones).all(), f'Expected finite value for ones tensos, got {srsim_ones}'
    srsim_zeros_ones = srsim(zeros, ones, data_range=1.)
    assert torch.isfinite(srsim_zeros_ones).all(), \
        f'Expected finite value for zeros and ones tensos, got {srsim_zeros_ones}'


def test_ssim_raises_if_bigger_kernel(device: str) -> None:
    # kernels bigger than image * scale
    prediction = torch.rand(1, 3, 50, 50, device=device)
    target = torch.rand(1, 3, 50, 50, device=device)
    with pytest.raises(ValueError):
        srsim(prediction, target, kernel_size=15)
    with pytest.raises(ValueError):
        srsim(prediction, target, gaussian_size=15)
    assert torch.isfinite(srsim(prediction, target, kernel_size=15, scale=0.5)).all()
    assert torch.isfinite(srsim(prediction, target, gaussian_size=15, scale=0.5)).all()
    assert torch.isfinite(srsim(prediction, target)).all()


def test_srsim_supports_different_data_ranges(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction, target = input_tensors
    prediction_255 = (prediction * 255).type(torch.uint8)
    target_255 = (target * 255).type(torch.uint8)
    measure_255 = srsim(prediction_255.to(device), target_255.to(device), data_range=255)
    measure = srsim((prediction_255 / 255.).to(device), (target_255 / 255.).to(device))

    diff = torch.abs(measure_255 - measure)
    assert diff <= 1e-6, f'Result for same tensor with different data_range should be the same, got {diff}'


def test_srsim_modes(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction, target = input_tensors
    for reduction in ['mean', 'sum', 'none']:
        srsim(prediction.to(device), target.to(device), reduction=reduction)

    for reduction in ['DEADBEEF', 'random']:
        with pytest.raises(ValueError):
            srsim(prediction.to(device), target.to(device), reduction=reduction)


def test_srsim_compare_with_matlab(device: str) -> None:
    # Greyscale image
    prediction = torch.tensor(imread('tests/assets/goldhill.gif')).unsqueeze(0).unsqueeze(0)
    target = torch.tensor(imread('tests/assets/goldhill_jpeg.gif')).unsqueeze(0).unsqueeze(0)
    # odd kernel (exactly same as matlab)
    predicted_score = srsim(prediction.to(device), target.to(device), gaussian_size=9, data_range=255, reduction='none')
    target_score = torch.tensor([0.94623509]).to(predicted_score)  # from matlab code
    assert torch.allclose(predicted_score, target_score), f'Expected MATLAB result {target_score.item():.8f},' \
                                                          f'got {predicted_score.item():.8f}'
    # even kernel (a bit different as matlab)
    predicted_score = srsim(prediction.to(device), target.to(device), data_range=255, reduction='none')
    target_score = torch.tensor([0.94652679]).to(predicted_score)  # from matlab code
    assert torch.allclose(predicted_score, target_score),\
        f'Expected MATLAB result {target_score.item():.8f}, got {predicted_score.item():.8f}'

    # RBG image
    prediction = torch.tensor(imread('tests/assets/I01.BMP')).permute(2, 0, 1).unsqueeze(0)
    target = torch.tensor(imread('tests/assets/i01_01_5.bmp')).permute(2, 0, 1).unsqueeze(0)
    # odd kernel (exactly same as matlab)
    predicted_score = srsim(prediction.to(device), target.to(device), gaussian_size=9, data_range=255, reduction='none')
    target_score = torch.tensor([0.96667468]).to(predicted_score)  # from matlab code
    assert torch.allclose(predicted_score, target_score), \
        f'Expected MATLAB result {target_score.item():.8f}, got {predicted_score.item():.8f}'
    # even kernel (a bit different as matlab)
    predicted_score = srsim(prediction.to(device), target.to(device), data_range=255, reduction='none')
    target_score = torch.tensor([0.9659730]).to(predicted_score)  # from matlab code
    assert torch.allclose(predicted_score, target_score), \
        f'Expected MATLAB result {target_score.item():.8f}, got {predicted_score.item():.8f}'


def test_srsim_chromatic(device: str) -> None:
    # Greyscale image
    prediction = torch.tensor(imread('tests/assets/goldhill.gif')).unsqueeze(0).unsqueeze(0)
    target = torch.tensor(imread('tests/assets/goldhill_jpeg.gif')).unsqueeze(0).unsqueeze(0)
    with pytest.raises(ValueError):
        srsim(prediction.to(device), target.to(device), data_range=255, chromatic=True, reduction='none')
    # RBG image
    prediction = torch.tensor(imread('tests/assets/I01.BMP')).permute(2, 0, 1).unsqueeze(0)
    target = torch.tensor(imread('tests/assets/i01_01_5.bmp')).permute(2, 0, 1).unsqueeze(0)
    predicted_score = srsim(prediction.to(device), target.to(device), data_range=255, chromatic=True, reduction='none')
    target_score = torch.tensor([0.9546513]).to(predicted_score)
    assert torch.allclose(predicted_score, target_score), f'Expected result for chromatic version,' \
                                                          f'got diff{predicted_score - target_score}'


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64],
)
def test_srsim_preserves_dtype(input_tensors: Tuple[torch.Tensor, torch.Tensor], dtype, device: str) -> None:
    x, y = input_tensors
    output = srsim(x.to(device=device, dtype=dtype), y.to(device=device, dtype=dtype))
    assert output.dtype == dtype


# ================== Test class: `srsimLoss` =================
def test_srsim_loss(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction, target = input_tensors
    prediction.requires_grad_()
    loss = SRSIMLoss(data_range=1.)(prediction.to(device), target.to(device))
    loss.backward()
    assert torch.isfinite(prediction.grad).all(), \
        f'Expected finite gradient values after back propagation, got {prediction.grad}'


def test_srsim_loss_zero_for_equal_input(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction, _ = input_tensors
    target = prediction.clone()
    prediction.requires_grad_()
    loss = SRSIMLoss(data_range=1.)(prediction.to(device), target.to(device))
    assert torch.isclose(loss, torch.zeros_like(loss)), \
        f'Expected loss equals zero for identical inputs, got {loss}'
