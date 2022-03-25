import torch
import pytest
from libsvm import svmutil  # noqa: F401
from brisque import BRISQUE
from piq import brisque, BRISQUELoss
from skimage.io import imread
from typing import Any


@pytest.fixture(scope='module')
def x_grey() -> torch.Tensor:
    return torch.rand(3, 1, 96, 96)


@pytest.fixture(scope='module')
def x_rgb() -> torch.Tensor:
    return torch.rand(3, 3, 96, 96)


# ================== Test function: `brisque` ==================
def test_brisque_works_with_grey(x_grey: torch.Tensor, device: str) -> None:
    brisque(x_grey.to(device))


def test_brisque_works_with_rgb(x_rgb, device: str) -> None:
    brisque(x_rgb.to(device))


def test_brisque_raises_if_wrong_reduction(x_grey: torch.Tensor, device: str) -> None:
    for mode in ['mean', 'sum', 'none']:
        brisque(x_grey.to(device), reduction=mode)

    for mode in [None, 'n', 2]:
        with pytest.raises(ValueError):
            brisque(x_grey.to(device), reduction=mode)


def test_brisque_values_grey(device: str) -> None:
    img = imread('tests/assets/goldhill.gif')
    x_grey = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    score = brisque(x_grey.to(device), reduction='none', data_range=255)
    score_baseline = BRISQUE().get_score(img)
    assert torch.isclose(score, torch.tensor(score_baseline).to(score), rtol=1e-3), \
        f'Expected values to be equal to baseline, got {score.item()} and {score_baseline}'


def test_brisque_values_rgb(device: str) -> None:
    img = imread('tests/assets/I01.BMP')
    x_rgb = (torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0))
    score = brisque(x_rgb.to(device), reduction='none', data_range=255.)
    score_baseline = BRISQUE().get_score(x_rgb[0].permute(1, 2, 0).numpy()[..., ::-1])
    assert torch.isclose(score, torch.tensor(score_baseline).to(score), rtol=1e-3), \
        f'Expected values to be equal to baseline, got {score.item()} and {score_baseline}'


@pytest.mark.parametrize(
    "input,expectation",
    [(torch.zeros(2, 1, 96, 96), pytest.raises(AssertionError)),
     (torch.ones(2, 1, 96, 96), pytest.raises(AssertionError))],
)
def test_brisque_for_special_cases(input: torch.Tensor, expectation: Any, device: str) -> None:
    with expectation:
        brisque(input.to(device), reduction='mean')


@pytest.mark.parametrize(
    "data_range", [128, 255],
)
def test_brisque_supports_different_data_ranges(x_rgb: torch.Tensor, data_range, device: str) -> None:
    x_scaled = (x_rgb * data_range).type(torch.uint8)
    loss_scaled = brisque(x_scaled.to(device), data_range=data_range)
    loss = brisque(x_scaled.to(device) / float(data_range), data_range=1.0)
    diff = torch.abs(loss_scaled - loss)
    assert diff <= 1e-5, f'Result for same tensor with different data_range should be the same, got {diff}'


def test_brisque_fails_for_incorrect_data_range(x_rgb: torch.Tensor, device: str) -> None:
    # Scale to [0, 255]
    x_scaled = (x_rgb * 255).type(torch.uint8)
    with pytest.raises(AssertionError):
        brisque(x_scaled.to(device), data_range=1.0)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64],
)
def test_brisque_preserves_dtype(input_tensors: torch.Tensor, dtype, device: str) -> None:
    x, _ = input_tensors
    output = brisque(x.to(device=device, dtype=dtype))
    assert output.dtype == dtype


# ================== Test class: `BRISQUELoss` ==================
def test_brisque_loss_if_works_with_grey(x_grey: torch.Tensor, device: str) -> None:
    x_grey_grad = x_grey.clone().to(device)
    x_grey_grad.requires_grad_()
    loss_value = BRISQUELoss()(x_grey_grad)
    loss_value.backward()
    assert torch.isfinite(x_grey_grad.grad).all(), \
        f'Expected non None gradient of leaf variable, got {x_grey_grad.grad}'


def test_brisque_loss_if_works_with_rgb(x_rgb: torch.Tensor, device: str) -> None:
    x_rgb_grad = x_rgb.clone().to(device)
    x_rgb_grad.requires_grad_()
    loss_value = BRISQUELoss()(x_rgb_grad)
    loss_value.backward()
    assert torch.isfinite(x_rgb_grad.grad).all(), \
        f'Expected non None gradient of leaf variable, got {x_rgb_grad.grad}'


def test_brisque_loss_raises_if_wrong_reduction(x_grey: torch.Tensor, device: str) -> None:
    for mode in ['mean', 'sum', 'none']:
        BRISQUELoss(reduction=mode)(x_grey.to(device))

    for mode in [None, 'n', 2]:
        with pytest.raises(ValueError):
            BRISQUELoss(reduction=mode)(x_grey.to(device))
