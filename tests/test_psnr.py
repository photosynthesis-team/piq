import torch
import pytest
from typing import Tuple
from skimage.metrics import peak_signal_noise_ratio

from piq import psnr


@pytest.fixture(scope='module')
def x() -> torch.Tensor:
    return torch.rand(4, 3, 128, 128)


@pytest.fixture(scope='module')
def y() -> torch.Tensor:
    return torch.rand(4, 3, 128, 128)


# ================== Test function: `psnr` ==================
def test_psnr(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    x, y = input_tensors
    psnr(x.to(device), y.to(device), data_range=1.0)


@pytest.mark.parametrize(
    "data_range", [128, 255],
)
def test_psnr_supports_different_data_ranges(
        input_tensors: Tuple[torch.Tensor, torch.Tensor], data_range, device: str) -> None:
    x, y = input_tensors
    x_scaled = (x * data_range).type(torch.uint8)
    y_scaled = (y * data_range).type(torch.uint8)

    measure_scaled = psnr(x_scaled.to(device), y_scaled.to(device), data_range=data_range)
    measure = psnr(
        x_scaled.to(device) / float(data_range),
        y_scaled.to(device) / float(data_range),
        data_range=1.0
    )
    diff = torch.abs(measure_scaled - measure)
    assert diff <= 1e-6, f'Result for same tensor with different data_range should be the same, got {diff}'


def test_psnr_fails_for_incorrect_data_range(x, y, device: str) -> None:
    # Scale to [0, 255]
    x_scaled = (x * 255).type(torch.uint8)
    y_scaled = (y * 255).type(torch.uint8)
    with pytest.raises(AssertionError):
        psnr(x_scaled.to(device), y_scaled.to(device), data_range=1.0)


def test_psnr_works_for_zero_tensors() -> None:
    x = torch.zeros(4, 3, 256, 256)
    y = torch.zeros(4, 3, 256, 256)
    measure = psnr(x, y, data_range=1.0)
    assert torch.isclose(measure, torch.tensor(80.))


def test_psnr_big_for_identical_images(x) -> None:
    # Max value depends on EPS constant. It's 80 for EPS=1e-8, 100 for EPS=1e-10 etc.
    max_val = torch.tensor(80.)

    y = x.clone()
    measure = psnr(x, y, data_range=1.0)
    assert torch.isclose(measure, max_val), f"PSNR for identical images should be 80, got {measure}"


def test_psnr_reduction(x, y):
    measure = psnr(x, y, reduction='mean')
    assert measure.dim() == 0, f'PSNR with `mean` reduction must return 1 number, got {len(measure)}'

    measure = psnr(x, y, reduction='sum')
    assert measure.dim() == 0, f'PSNR with `mean` reduction must return 1 number, got {len(measure)}'

    measure = psnr(x, y, reduction='none')
    assert len(measure) == x.size(0), \
        f'PSNR with `none` reduction must have length equal to number of images, got {len(measure)}'

    with pytest.raises(ValueError):
        psnr(x, y, reduction='random string')


def test_psnr_matches_skimage_greyscale():
    x = torch.rand(1, 1, 256, 256)
    y = torch.rand(1, 1, 256, 256)
    pm_measure = psnr(x, y, reduction='mean')
    sk_measure = peak_signal_noise_ratio(x.squeeze().numpy(), y.squeeze().numpy(), data_range=1.0)

    assert torch.isclose(pm_measure, torch.tensor(sk_measure, dtype=pm_measure.dtype)), \
        f"Must match Sklearn version. Got: {pm_measure} and skimage: {sk_measure}"


def test_psnr_matches_skimage_rgb():
    x = torch.rand(1, 3, 256, 256)
    y = torch.rand(1, 3, 256, 256)
    pm_measure = psnr(x, y, reduction='mean')
    sk_measure = peak_signal_noise_ratio(x.squeeze().numpy(), y.squeeze().numpy(), data_range=1.0)

    assert torch.isclose(pm_measure, torch.tensor(sk_measure, dtype=pm_measure.dtype)), \
        f"Must match Sklearn version. Got: {pm_measure} and skimage: {sk_measure}"


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64],
)
def test_psnr_preserves_dtype(x, y, dtype, device: str) -> None:
    output = psnr(x.to(device=device, dtype=dtype), y.to(device=device, dtype=dtype))
    assert output.dtype == dtype


def test_psnr_loss_backward():
    x = torch.rand(1, 3, 256, 256, requires_grad=True)
    y = torch.rand(1, 3, 256, 256)
    loss = 80 - psnr(x, y, reduction='mean')
    loss.backward()
    assert x.grad is not None, 'Expected non None gradient of leaf variable'
