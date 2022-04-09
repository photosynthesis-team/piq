import torch
import pytest
from typing import Any, Tuple
from contextlib import contextmanager

from skimage.io import imread
from piq import LPIPS


@contextmanager
def raise_nothing():
    yield


# ================== Test class: `LPIPS` ==================
def test_lpips_loss_init() -> None:
    LPIPS()


def test_lpips_loss_forward(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    x, y = input_tensors
    loss = LPIPS()
    loss(x.to(device), y.to(device))


def test_lpips_computes_grad(x, y, device: str) -> None:
    x.requires_grad_()
    loss_value = LPIPS(enable_grad=True)(x.to(device), y.to(device))
    loss_value.backward()
    assert x.grad is not None, 'Expected non None gradient of leaf variable'


def test_lpips_loss_does_not_compute_grad(x, y, device: str) -> None:
    x.requires_grad_()
    loss_value = LPIPS(enable_grad=False)(x.to(device), y.to(device))
    with pytest.raises(RuntimeError):
        loss_value.backward()


def test_lpips_loss_raises_if_wrong_reduction(x, y) -> None:
    for mode in ['mean', 'sum', 'none']:
        LPIPS(reduction=mode)(x, y)

    for mode in [None, 'n', 2]:
        with pytest.raises(ValueError):
            LPIPS(reduction=mode)(x, y)


@pytest.mark.parametrize(
    "x, y, expectation, value",
    [
        (torch.zeros(2, 3, 96, 96), torch.zeros(2, 3, 96, 96), raise_nothing(), 0.0),
        (torch.ones(2, 3, 96, 96), torch.ones(2, 3, 96, 96), raise_nothing(), 0.0),
    ],
)
def test_lpips_loss_forward_for_special_cases(x, y, expectation: Any, value: float) -> None:
    loss = LPIPS()
    with expectation:
        loss_value = loss(x, y)
        assert torch.isclose(loss_value, torch.tensor(value), atol=1e-6), \
            f'Expected loss value to be equal to target value. Got {loss_value} and {value}'


def test_lpips_similar_to_official_implementation():
    """Used official LPIPS code from https://github.com/richzhang/PerceptualSimilarity
    Evaluated "vgg" model with v0.1 weights and lpips_2imgs.py script.
    """
    # RGB images
    I01 = torch.tensor(imread('tests/assets/I01.BMP')).permute(2, 0, 1)[None, ...]
    i1_01_5 = torch.tensor(imread('tests/assets/i01_01_5.bmp')).permute(2, 0, 1)[None, ...]

    lpips_metric = LPIPS(enable_grad=False, reduction='none', data_range=255)
    score = lpips_metric(i1_01_5, I01)

    # Baseline values are from original PyTorch code
    score_baseline = torch.tensor(0.605)

    assert torch.isclose(score, score_baseline), \
        f'Expected PIQ score to be equal to PyTorch prediction. Got {score} and {score_baseline}'
