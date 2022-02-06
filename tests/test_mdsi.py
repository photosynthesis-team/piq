import torch
import pytest
from skimage.io import imread
from piq import mdsi, MDSILoss
from typing import Tuple, Any

x_image = [
    torch.tensor(imread('tests/assets/goldhill_jpeg.gif'), dtype=torch.float32).unsqueeze(0).unsqueeze(0),
    torch.tensor(imread('tests/assets/i01_01_5.bmp'), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
]

y_image = [
    torch.tensor(imread('tests/assets/goldhill.gif'), dtype=torch.float32).unsqueeze(0).unsqueeze(0),
    torch.tensor(imread('tests/assets/I01.BMP'), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
]
y_score = [
    {'sum': torch.tensor(0.395910876130775), 'mult': torch.tensor(0.328692181289061)},
    {'sum': torch.tensor(0.338972600511665), 'mult': torch.tensor(0.258002917257516)}
]


@pytest.fixture(params=zip(x_image, y_image, y_score))
def input_images_score(request: Any) -> Any:
    return request.param


# ================== Test function: `mdsi` ==================
def test_mdsi(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    x, y = input_tensors
    score = mdsi(x=x.to(device), y=y.to(device), data_range=1., reduction='none')
    assert torch.isfinite(score).all(), f'Expected finite scores, got {score}'


def test_mdsi_reduction(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    x, y = input_tensors
    for reduction in ['mean', 'sum', 'none']:
        mdsi(x=x.to(device), y=y.to(device), data_range=1., reduction=reduction)


def test_mdsi_combination(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    x, y = input_tensors
    for combination in ['sum', 'mult']:
        mdsi(x=x.to(device), y=y.to(device), data_range=1., combination=combination)
    for combination in ['DEADBEEF', 'random']:
        with pytest.raises(ValueError):
            mdsi(x=x.to(device), y=y.to(device), data_range=1., combination=combination)


@pytest.mark.parametrize(
    "data_range", [128, 255],
)
def test_mdsi_supports_different_data_ranges(
        input_tensors: Tuple[torch.Tensor, torch.Tensor], data_range, device: str) -> None:
    x, y = input_tensors
    x_scaled = (x * data_range).type(torch.uint8)
    y_scaled = (y * data_range).type(torch.uint8)

    measure_scaled = mdsi(x_scaled.to(device), y_scaled.to(device), data_range=data_range)
    measure = mdsi(
        x_scaled.to(device) / float(data_range),
        y_scaled.to(device) / float(data_range),
        data_range=1.0
    )
    diff = torch.abs(measure_scaled - measure)
    assert diff <= 1e-6, f'Result for same tensor with different data_range should be the same, got {diff}'


def test_mdsi_fails_for_incorrect_data_range(x, y, device: str) -> None:
    # Scale to [0, 255]
    x_scaled = (x * 255).type(torch.uint8)
    y_scaled = (y * 255).type(torch.uint8)
    with pytest.raises(AssertionError):
        mdsi(x_scaled.to(device), y_scaled.to(device), data_range=1.0)


@pytest.mark.parametrize("combination", ['sum', 'mult'])
def test_mdsi_compare_with_matlab(input_images_score: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                  combination: str, device: str) -> None:
    x, y, y_value = input_images_score
    y_value = y_value[combination]
    score = mdsi(x=x.to(device), y=y.to(device), data_range=255, combination=combination)
    assert torch.isclose(score, y_value.to(score)), f'The estimated value must be equal to MATLAB provided one, ' \
                                                    f'got {score.item():.8f}, while MATLAB equals {y_value}'


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64],
)
def test_mdsi_preserves_dtype(x, y, dtype, device: str) -> None:
    output = mdsi(x.to(device=device, dtype=dtype), y.to(device=device, dtype=dtype))
    assert output.dtype == dtype


# ================== Test function: `MDSILoss` ==================
def test_mdsi_loss(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    x, y = input_tensors
    x.requires_grad_()
    loss = MDSILoss(data_range=1.)(x=x.to(device), y=y.to(device))
    loss.backward()
    assert torch.isfinite(x.grad).all(), f'Expected finite gradient values, got {x.grad}'


@pytest.mark.parametrize("combination", ['sum', 'mult'])
def test_mdsi_loss_compare_with_matlab(input_images_score: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                       combination: str, device: str) -> None:
    x, y, y_value = input_images_score
    y_value = y_value[combination]
    x = x.requires_grad_()
    score = MDSILoss(data_range=255, combination=combination)(x=x.to(device), y=y.to(device))
    score.backward()
    assert torch.isclose(score, y_value.to(score)), f'The estimated value must be equal to MATLAB ' \
                                                    f'provided one, got {score.item():.8f}, ' \
                                                    f'while MATLAB equals {y_value}'
    assert torch.isfinite(x.grad).all(), f'Expected finite gradient values, got {x.grad}'
