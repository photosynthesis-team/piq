import torch
import pytest
from typing import Tuple
from skimage.metrics import peak_signal_noise_ratio

from piq import psnr


@pytest.fixture(scope='module')
def prediction() -> torch.Tensor:
    return torch.rand(4, 3, 128, 128)


@pytest.fixture(scope='module')
def target() -> torch.Tensor:
    return torch.rand(4, 3, 128, 128)


# ================== Test function: `psnr` ==================
def test_psnr(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction, target = input_tensors
    psnr(prediction.to(device), target.to(device), data_range=1.0)


@pytest.mark.parametrize(
    "data_range", [128, 255],
)
def test_psnr_supports_different_data_ranges(
        input_tensors: Tuple[torch.Tensor, torch.Tensor], data_range, device: str) -> None:
    prediction, target = input_tensors
    prediction_scaled = (prediction * data_range).type(torch.uint8)
    target_scaled = (target * data_range).type(torch.uint8)

    measure_scaled = psnr(prediction_scaled.to(device), target_scaled.to(device), data_range=data_range)
    measure = psnr(
        prediction_scaled.to(device) / float(data_range),
        target_scaled.to(device) / float(data_range),
        data_range=1.0
    )
    diff = torch.abs(measure_scaled - measure)
    assert diff <= 1e-6, f'Result for same tensor with different data_range should be the same, got {diff}'


def test_psnr_fails_for_incorrect_data_range(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    # Scale to [0, 255]
    prediction_scaled = (prediction * 255).type(torch.uint8)
    target_scaled = (target * 255).type(torch.uint8)
    with pytest.raises(AssertionError):
        psnr(prediction_scaled.to(device), target_scaled.to(device), data_range=1.0)
        
        
def test_psnr_works_for_zero_tensors() -> None:
    prediction = torch.zeros(4, 3, 256, 256)
    target = torch.zeros(4, 3, 256, 256)
    measure = psnr(prediction, target, data_range=1.0)
    assert torch.isclose(measure, torch.tensor(80.))


def test_psnr_big_for_identical_images(prediction: torch.Tensor) -> None:
    # Max value depends on EPS constant. It's 80 for EPS=1e-8, 100 for EPS=1e-10 etc.
    max_val = torch.tensor(80.)

    target = prediction.clone()
    measure = psnr(prediction, target, data_range=1.0)
    assert torch.isclose(measure, max_val), f"PSNR for identical images should be 80, got {measure}"


def test_psnr_reduction(prediction: torch.Tensor, target: torch.Tensor):
    measure = psnr(prediction, target, reduction='mean')
    assert measure.dim() == 0, f'PSNR with `mean` reduction must return 1 number, got {len(measure)}'

    measure = psnr(prediction, target, reduction='sum')
    assert measure.dim() == 0, f'PSNR with `mean` reduction must return 1 number, got {len(measure)}'

    measure = psnr(prediction, target, reduction='none')
    assert len(measure) == prediction.size(0), \
        f'PSNR with `none` reduction must have length equal to number of images, got {len(measure)}'

    with pytest.raises(KeyError):
        psnr(prediction, target, reduction='random string')


def test_psnr_matches_skimage_greyscale():
    prediction = torch.rand(1, 1, 256, 256)
    target = torch.rand(1, 1, 256, 256)
    pm_measure = psnr(prediction, target, reduction='mean')
    sk_measure = peak_signal_noise_ratio(prediction.squeeze().numpy(), target.squeeze().numpy(), data_range=1.0)

    assert torch.isclose(pm_measure, torch.tensor(sk_measure, dtype=pm_measure.dtype)), \
        f"Must match Sklearn version. Got: {pm_measure} and skimage: {sk_measure}"


def test_psnr_matches_skimage_rgb():
    prediction = torch.rand(1, 3, 256, 256)
    target = torch.rand(1, 3, 256, 256)
    pm_measure = psnr(prediction, target, reduction='mean')
    sk_measure = peak_signal_noise_ratio(prediction.squeeze().numpy(), target.squeeze().numpy(), data_range=1.0)

    assert torch.isclose(pm_measure, torch.tensor(sk_measure, dtype=pm_measure.dtype)), \
        f"Must match Sklearn version. Got: {pm_measure} and skimage: {sk_measure}"


def test_psnr_loss_backward():
    prediction = torch.rand(1, 3, 256, 256, requires_grad=True)
    target = torch.rand(1, 3, 256, 256)
    loss = 80 - psnr(prediction, target, reduction='mean')
    loss.backward()
    assert prediction.grad is not None, 'Expected non None gradient of leaf variable'
