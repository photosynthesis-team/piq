import torch
import pytest
from libsvm import svmutil  # noqa: F401
from brisque import BRISQUE
from piq import brisque, BRISQUELoss
from PIL import Image
import numpy as np
from typing import Any


@pytest.fixture(scope='module')
def prediction_grey() -> torch.Tensor:
    return torch.rand(3, 1, 128, 128)


@pytest.fixture(scope='module')
def prediction_rgb() -> torch.Tensor:
    return torch.rand(3, 3, 128, 128)


# ================== Test function: `brisque` ==================
def test_brisque_if_works_with_grey(prediction_grey: torch.Tensor, device: Any) -> None:
    brisque(prediction_grey.to(device))


def test_brisque_if_works_with_rgb(prediction_rgb, device: Any) -> None:
    brisque(prediction_rgb)


def test_brisque_raises_if_wrong_reduction(prediction_grey: torch.Tensor, device: Any) -> None:
    for mode in ['mean', 'sum', 'none']:
        brisque(prediction_grey.to(device), reduction=mode)

    for mode in [None, 'n', 2]:
        with pytest.raises(KeyError):
            brisque(prediction_grey.to(device), reduction=mode)


def test_brisque_values_grey(device: Any) -> None:
    img = np.array(Image.open('tests/assets/goldhill.gif'))
    prediction_grey = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    score = brisque(prediction_grey.to(device), reduction='none', data_range=255)
    score_baseline = BRISQUE().get_score(img)
    assert torch.isclose(score, torch.tensor(score_baseline).to(score), rtol=1e-3), \
        f'Expected values to be equal to baseline prediction, got {score.item()} and {score_baseline}'


def test_brisque_values_rgb(device: Any) -> None:
    img = np.array(Image.open('tests/assets/I01.BMP'))
    prediction_rgb = (torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0))
    score = brisque(prediction_rgb.to(device), reduction='none', data_range=255.)
    score_baseline = BRISQUE().get_score(prediction_rgb[0].permute(1, 2, 0).numpy()[..., ::-1])
    assert torch.isclose(score, torch.tensor(score_baseline).to(score), rtol=1e-3), \
        f'Expected values to be equal to baseline prediction, got {score.item()} and {score_baseline}'


def test_brisque_all_zeros(device: Any) -> None:
    size = (1, 1, 128, 128)
    for tensor in [torch.zeros(size), torch.ones(size)]:
        with pytest.raises(AssertionError):
            brisque(tensor.to(device), reduction='mean')


# ================== Test class: `BRISQUELoss` ==================
def test_brisque_loss_if_works_with_grey(prediction_grey: torch.Tensor, device: Any) -> None:
    prediction_grey_grad = prediction_grey.clone().to(device)
    prediction_grey_grad.requires_grad_()
    loss_value = BRISQUELoss()(prediction_grey_grad)
    loss_value.backward()
    assert torch.isfinite(prediction_grey_grad.grad).all(), f'Expected non None gradient of leaf variable, ' \
                                                            f'got {prediction_grey_grad.grad}'


def test_brisque_loss_if_works_with_rgb(prediction_rgb: torch.Tensor, device: Any) -> None:
    prediction_rgb_grad = prediction_rgb.clone().to(device)
    prediction_rgb_grad.requires_grad_()
    loss_value = BRISQUELoss()(prediction_rgb_grad)
    loss_value.backward()
    assert torch.isfinite(prediction_rgb_grad.grad).all(), 'Expected non None gradient of leaf variable, ' \
                                                           f'got {prediction_rgb_grad.grad}'


def test_brisque_loss_raises_if_wrong_reduction(prediction_grey: torch.Tensor, device: Any) -> None:
    for mode in ['mean', 'sum', 'none']:
        BRISQUELoss(reduction=mode)(prediction_grey.to(device))

    for mode in [None, 'n', 2]:
        with pytest.raises(KeyError):
            BRISQUELoss(reduction=mode)(prediction_grey.to(device))
