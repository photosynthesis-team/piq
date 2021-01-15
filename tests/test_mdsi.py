import torch
import pytest
from skimage.io import imread
from piq import mdsi, MDSILoss
from typing import Tuple, Any

prediction_image = [
    torch.tensor(imread('tests/assets/goldhill_jpeg.gif'), dtype=torch.float32).unsqueeze(0).unsqueeze(0),
    torch.tensor(imread('tests/assets/i01_01_5.bmp'), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
]

target_image = [
    torch.tensor(imread('tests/assets/goldhill.gif'), dtype=torch.float32).unsqueeze(0).unsqueeze(0),
    torch.tensor(imread('tests/assets/I01.BMP'), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
]
target_score = [
    {'sum': torch.tensor(0.395910876130775), 'mult': torch.tensor(0.328692181289061)},
    {'sum': torch.tensor(0.338972600511665), 'mult': torch.tensor(0.258002917257516)}
]


@pytest.fixture(params=zip(prediction_image, target_image, target_score))
def input_images_score(request: Any) -> Any:
    return request.param


# ================== Test function: `mdsi` ==================
def test_mdsi(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction, target = input_tensors
    score = mdsi(prediction=prediction.to(device), target=target.to(device), data_range=1., reduction='none')
    assert torch.isfinite(score).all(), f'Expected finite scores, got {score}'


def test_mdsi_reduction(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction, target = input_tensors
    for reduction in ['mean', 'sum', 'none']:
        mdsi(prediction=prediction.to(device), target=target.to(device), data_range=1., reduction=reduction)


def test_mdsi_combination(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction, target = input_tensors
    for combination in ['sum', 'mult']:
        mdsi(prediction=prediction.to(device), target=target.to(device), data_range=1., combination=combination)
    for combination in ['DEADBEEF', 'random']:
        with pytest.raises(ValueError):
            mdsi(prediction=prediction.to(device), target=target.to(device), data_range=1., combination=combination)


@pytest.mark.parametrize(
    "data_range", [128, 255],
)
def test_mdsi_supports_different_data_ranges(
        input_tensors: Tuple[torch.Tensor, torch.Tensor], data_range, device: str) -> None:
    prediction, target = input_tensors
    prediction_scaled = (prediction * data_range).type(torch.uint8)
    target_scaled = (target * data_range).type(torch.uint8)

    measure_scaled = mdsi(prediction_scaled.to(device), target_scaled.to(device), data_range=data_range)
    measure = mdsi(
        prediction_scaled.to(device) / float(data_range),
        target_scaled.to(device) / float(data_range),
        data_range=1.0
    )
    diff = torch.abs(measure_scaled - measure)
    assert diff <= 1e-6, f'Result for same tensor with different data_range should be the same, got {diff}'


def test_mdsi_fails_for_incorrect_data_range(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    # Scale to [0, 255]
    prediction_scaled = (prediction * 255).type(torch.uint8)
    target_scaled = (target * 255).type(torch.uint8)
    with pytest.raises(AssertionError):
        mdsi(prediction_scaled.to(device), target_scaled.to(device), data_range=1.0)
        

@pytest.mark.parametrize("combination", ['sum', 'mult'])
def test_mdsi_compare_with_matlab(input_images_score: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                  combination: str, device: str) -> None:
    prediction, target, target_value = input_images_score
    target_value = target_value[combination]
    score = mdsi(prediction=prediction.to(device), target=target.to(device), data_range=255, combination=combination)
    assert torch.isclose(score, target_value.to(score)), f'The estimated value must be equal to MATLAB provided one, ' \
                                                         f'got {score.item():.8f}, while MATLAB equals {target_value}'


# ================== Test function: `MDSILoss` ==================
def test_mdsi_loss(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction, target = input_tensors
    prediction.requires_grad_()
    loss = MDSILoss(data_range=1.)(prediction=prediction.to(device), target=target.to(device))
    loss.backward()
    assert torch.isfinite(prediction.grad).all(), f'Expected finite gradient values, got {prediction.grad}'


@pytest.mark.parametrize("combination", ['sum', 'mult'])
def test_mdsi_loss_compare_with_matlab(input_images_score: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                       combination: str, device: str) -> None:
    prediction, target, target_value = input_images_score
    target_value = target_value[combination]
    prediction = prediction.requires_grad_()
    score = MDSILoss(data_range=255, combination=combination)(prediction=prediction.to(device),
                                                              target=target.to(device))
    score.backward()
    assert torch.isclose(score, 1. - target_value.to(score)), f'The estimated value must be equal to MATLAB ' \
                                                              f'provided one, got {score.item():.8f}, ' \
                                                              f'while MATLAB equals {1. - target_value}'
    assert torch.isfinite(prediction.grad).all(), f'Expected finite gradient values, got {prediction.grad}'
