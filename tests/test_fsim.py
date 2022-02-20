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
def x() -> torch.Tensor:
    return torch.rand(3, 3, 256, 256)


@pytest.fixture(scope='module')
def y() -> torch.Tensor:
    return torch.rand(3, 3, 256, 256)


@pytest.fixture(scope='module')
def x_grey() -> torch.Tensor:
    return torch.rand(3, 1, 256, 256)


@pytest.fixture(scope='module')
def y_grey() -> torch.Tensor:
    return torch.rand(3, 1, 256, 256)


# ================== Test function: `fsim` ==================
def test_fsim_forward(input_tensors, device: str) -> None:
    x, y = input_tensors
    fsim(x.to(device), y.to(device), chromatic=False)


@pytest.mark.parametrize("chromatic", [False, True])
def test_fsim_symmetry(x, y, chromatic: bool, device: str) -> None:
    measure = fsim(x.to(device), y.to(device), data_range=1., chromatic=chromatic)
    reverse_measure = fsim(y.to(device), x.to(device), data_range=1., chromatic=chromatic)
    assert (measure == reverse_measure).all(), f'Expect: FSIM(a, b) == FSIM(b, a), got {measure} != {reverse_measure}'


@pytest.mark.parametrize(
    "chromatic,expectation",
    [(False, raise_nothing()),
     (True, pytest.raises(AssertionError))])
def test_fsim_chromatic_raises_for_greyscale(x_grey, y_grey, chromatic: bool, expectation: Any) -> None:
    with expectation:
        fsim(x_grey, y_grey, data_range=1., chromatic=chromatic)


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


@pytest.mark.parametrize(
    "data_range", [128, 255],
)
def test_fsim_supports_different_data_ranges(x, y, data_range, device: str) -> None:
    x_scaled = (x * data_range).type(torch.uint8)
    y_scaled = (y * data_range).type(torch.uint8)
    measure_scaled = fsim(x_scaled.to(device), y_scaled.to(device), data_range=data_range)
    measure = fsim(
        x_scaled.to(device) / float(data_range),
        y_scaled.to(device) / float(data_range),
        data_range=1.0
    )
    diff = torch.abs(measure_scaled - measure)
    assert diff <= 1e-5, f'Result for same tensor with different data_range should be the same, got {diff}'


def test_fsim_fails_for_incorrect_data_range(x, y, device: str) -> None:
    # Scale to [0, 255]
    x_scaled = (x * 255).type(torch.uint8)
    y_scaled = (y * 255).type(torch.uint8)
    with pytest.raises(AssertionError):
        fsim(x_scaled.to(device), y_scaled.to(device), data_range=1.0)


def test_fsim_simmular_to_matlab_implementation():
    # Greyscale images
    goldhill = torch.tensor(imread('tests/assets/goldhill.gif'))[None, None, ...]
    goldhill_jpeg = torch.tensor(imread('tests/assets/goldhill_jpeg.gif'))[None, None, ...]

    score = fsim(goldhill_jpeg, goldhill, data_range=255, chromatic=False, reduction='none')
    score_baseline = torch.tensor(0.89691)

    assert torch.isclose(score, score_baseline), \
        f'Expected PyTorch score to be equal to MATLAB prediction. Got {score} and {score_baseline}'

    # RGB images
    I01 = torch.tensor(imread('tests/assets/I01.BMP')).permute(2, 0, 1)[None, ...]
    i1_01_5 = torch.tensor(imread('tests/assets/i01_01_5.bmp')).permute(2, 0, 1)[None, ...]

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


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64],
)
def test_fsim_preserves_dtype(input_tensors: torch.Tensor, dtype, device: str) -> None:
    x, y = input_tensors
    output = fsim(x.to(device=device, dtype=dtype), y.to(device=device, dtype=dtype), chromatic=False)
    assert output.dtype == dtype


# ================== Test class: `FSIMLoss` ==================
def test_fsim_loss_reduction(x, y) -> None:
    loss = FSIMLoss(reduction='mean')
    measure = loss(x, y)
    assert measure.dim() == 0, f'FSIM with `mean` reduction must return 1 number, got {len(measure)}'

    loss = FSIMLoss(reduction='sum')
    measure = loss(x, y)
    assert measure.dim() == 0, f'FSIM with `mean` reduction must return 1 number, got {len(measure)}'

    loss = FSIMLoss(reduction='none')
    measure = loss(x, y)
    assert len(measure) == x.size(0), \
        f'FSIM with `none` reduction must have length equal to number of images, got {len(measure)}'

    loss = FSIMLoss(reduction='random string')
    with pytest.raises(ValueError):
        loss(x, y)


def test_fsim_loss_computes_grad(x, y, device: str) -> None:
    x.requires_grad_()
    loss_value = FSIMLoss()(x.to(device), y.to(device))
    loss_value.backward()
    assert x.grad is not None, 'Expected non None gradient of leaf variable'
