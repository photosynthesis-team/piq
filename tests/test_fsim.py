from typing import Any
import torch
import pytest
from skimage.io import imread
from contextlib import contextmanager

from piq import fsim, FSIMLoss


@contextmanager
def raise_nothing():
    yield


@pytest.fixture(scope='module')
def prediction() -> torch.Tensor:
    return torch.rand(3, 3, 256, 256)


@pytest.fixture(scope='module')
def target() -> torch.Tensor:
    return torch.rand(3, 3, 256, 256)


@pytest.fixture(scope='module')
def prediction_grey() -> torch.Tensor:
    return torch.rand(3, 3, 256, 256)


@pytest.fixture(scope='module')
def target_grey() -> torch.Tensor:
    return torch.rand(3, 3, 256, 256)


# ================== Test function: `fsim` ==================
@pytest.mark.parametrize("chromatic", [False, True])
def test_fsim_symmetry(prediction: torch.Tensor, target: torch.Tensor, chromatic: bool, device: str) -> None:
    measure = fsim(prediction, target, data_range=1., chromatic=chromatic)
    reverse_measure = fsim(target, prediction, data_range=1., chromatic=chromatic)
    assert (measure == reverse_measure).all(), f'Expect: FSIM(a, b) == FSIM(b, a), got {measure} != {reverse_measure}'


@pytest.mark.parametrize(
    "x,y,expectation,value",
    [
        (torch.rand(4, 3, 128, 128, 2), torch.rand(4, 3, 128, 128, 2), pytest.raises(AssertionError), None),
        (torch.randn(4, 3, 128, 128), torch.randn(4, 3, 128, 128), pytest.raises(AssertionError), None),
        (torch.zeros(4, 3, 128, 128), torch.zeros(4, 3, 128, 128), raise_nothing(), 1.0),
        (torch.ones(4, 3, 128, 128), torch.ones(4, 3, 128, 128), raise_nothing(), 1.0),
        (torch.rand(4, 3, 28, 28), torch.rand(4, 3, 28, 28), raise_nothing(), None),
    ],
)
def test_fsim_for_special_cases(x: torch.Tensor, y: torch.Tensor, expectation: Any, value: float) -> None:
    with expectation:
        if value is None:
            fsim(x, y)
        else:
            score = fsim(x, y)
            assert torch.isclose(score, torch.tensor(value)), \
                f'Expected loss value to be equal to target value. Got {score} and {value}'


def test_fsim_simmular_to_matlab_implementation():
    # Greyscale images
    goldhill = torch.tensor(imread('tests/assets/goldhill.gif'))
    goldhill_jpeg = torch.tensor(imread('tests/assets/goldhill_jpeg.gif'))

    score = fsim(goldhill_jpeg, goldhill, data_range=255, chromatic=False, reduction='none')
    score_baseline = torch.tensor(0.89691)

    assert torch.isclose(score, score_baseline), \
        f'Expected PyTorch score to be equal to MATLAB prediction. Got {score} and {score_baseline}'

    # RGB images
    I01 = torch.tensor(imread('tests/assets/I01.BMP')).permute(2, 0, 1)
    i1_01_5 = torch.tensor(imread('tests/assets/i01_01_5.bmp')).permute(2, 0, 1)

    score = fsim(i1_01_5, I01, data_range=255, chromatic=False, reduction='none')
    score_chromatic = fsim(i1_01_5, I01, data_range=255, chromatic=True, reduction='none')

    # Baseline values are from original MATLAB code
    score_baseline = torch.tensor(0.93674)
    score_baseline_chromatic = torch.tensor(0.92587)

    assert torch.isclose(score, score_baseline), \
        f'Expected PyTorch score to be equal to MATLAB prediction. Got {score} and {score_baseline}'
    assert torch.isclose(score_chromatic, score_baseline_chromatic, atol=1e-4), \
        'Expected PyTorch chromatic score to be equal to MATLAB prediction.' \
        f'Got {score_chromatic} and {score_baseline_chromatic}'


# ================== Test class: `FSIMLoss` ==================
def test_fsim_loss_reduction(prediction: torch.Tensor, target: torch.Tensor) -> None:
    loss = FSIMLoss(reduction='mean')
    measure = loss(prediction, target)
    assert measure.dim() == 0, f'FSIM with `mean` reduction must return 1 number, got {len(measure)}'

    loss = FSIMLoss(reduction='sum')
    measure = loss(prediction, target)
    assert measure.dim() == 0, f'FSIM with `mean` reduction must return 1 number, got {len(measure)}'

    loss = FSIMLoss(reduction='none')
    measure = loss(prediction, target)
    assert len(measure) == prediction.size(0), \
        f'FSIM with `none` reduction must have length equal to number of images, got {len(measure)}'
    
    loss = FSIMLoss(reduction='random string')
    with pytest.raises(KeyError):
        loss(prediction, target)


def test_fsim_loss_computes_grad(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    prediction.requires_grad_()
    loss_value = FSIMLoss()(prediction, target)
    loss_value.backward()
    assert prediction.grad is not None, 'Expected non None gradient of leaf variable'
