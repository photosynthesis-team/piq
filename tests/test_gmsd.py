import torch
import pytest
from skimage.io import imread
import numpy as np
from typing import Any, Tuple

from piq import gmsd, multi_scale_gmsd, GMSDLoss, MultiScaleGMSDLoss

LEAF_VARIABLE_ERROR_MESSAGE = 'Expected non None gradient of leaf variable'


@pytest.fixture(scope='module')
def prediction() -> torch.Tensor:
    return torch.rand(2, 3, 96, 96)


@pytest.fixture(scope='module')
def target() -> torch.Tensor:
    return torch.rand(2, 3, 96, 96)


prediction_image = [
    torch.tensor(imread('tests/assets/goldhill_jpeg.gif'), dtype=torch.float32).unsqueeze(0).unsqueeze(0),
    torch.tensor(imread('tests/assets/i01_01_5.bmp'), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
]

target_image = [
    torch.tensor(imread('tests/assets/goldhill.gif'), dtype=torch.float32).unsqueeze(0).unsqueeze(0),
    torch.tensor(imread('tests/assets/I01.BMP'), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
]
target_score = [
    torch.tensor(0.138012587141798),
    torch.tensor(0.094124655829098)
]


@pytest.fixture(params=zip(prediction_image, target_image, target_score))
def input_images_score(request: Any) -> Any:
    return request.param


# ================== Test function: `gmsd` ==================
def test_gmsd_forward(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    gmsd(prediction.to(device), target.to(device))


def test_gmsd_zero_for_equal_tensors(prediction: torch.Tensor, device: str) -> None:
    target = prediction.clone()
    measure = gmsd(prediction.to(device), target.to(device))
    assert measure.abs() <= 1e-6, f'GMSD for equal tensors must be 0, got {measure}'


def test_gmsd_raises_if_tensors_have_different_types(target: torch.Tensor, device: str) -> None:
    wrong_type_predictions = [list(range(10)), np.arange(10)]
    for wrong_type_prediction in wrong_type_predictions:
        with pytest.raises(AssertionError):
            gmsd(wrong_type_prediction, target.to(device))


@pytest.mark.parametrize(
    "data_range", [128, 255],
)
def test_gmsd_supports_different_data_ranges(
        prediction: torch.Tensor, target: torch.Tensor, data_range, device: str) -> None:
    prediction_scaled = (prediction * data_range).type(torch.uint8)
    target_scaled = (target * data_range).type(torch.uint8)
    measure_scaled = gmsd(prediction_scaled.to(device), target_scaled.to(device), data_range=data_range)
    measure = gmsd(
        prediction_scaled.to(device) / float(data_range),
        target_scaled.to(device) / float(data_range),
        data_range=1.0
    )
    diff = torch.abs(measure_scaled - measure)
    assert diff <= 1e-6, f'Result for same tensor with different data_range should be the same, got {diff}'


def test_gmsd_fails_for_incorrect_data_range(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    # Scale to [0, 255]
    prediction_scaled = (prediction * 255).type(torch.uint8)
    target_scaled = (target * 255).type(torch.uint8)
    with pytest.raises(AssertionError):
        gmsd(prediction_scaled.to(device), target_scaled.to(device), data_range=1.0)
        

def test_gmsd_supports_greyscale_tensors(device: str) -> None:
    target = torch.ones(2, 1, 96, 96)
    prediction = torch.zeros(2, 1, 96, 96)
    gmsd(prediction.to(device), target.to(device))


def test_gmsd_modes(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    for reduction in ['mean', 'sum', 'none']:
        gmsd(prediction.to(device), target.to(device), reduction=reduction)

    for reduction in ['DEADBEEF', 'random']:
        with pytest.raises(KeyError):
            gmsd(prediction.to(device), target.to(device), reduction=reduction)


def test_gmsd_compare_with_matlab(input_images_score: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                  device: str) -> None:
    prediction, target, target_value = input_images_score
    score = gmsd(prediction=prediction.to(device), target=target.to(device), data_range=255)
    assert torch.isclose(score, target_value.to(score)), f'The estimated value must be equal to MATLAB provided one, ' \
                                                         f'got {score.item():.8f}, while MATLAB equals {target_value}'


# ================== Test class: `GMSDLoss` ==================
def test_gmsd_loss_forward_backward(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    prediction.requires_grad_()
    loss_value = GMSDLoss()(prediction.to(device), target.to(device))
    loss_value.backward()
    assert torch.isfinite(prediction.grad).all(), LEAF_VARIABLE_ERROR_MESSAGE


def test_gmsd_loss_zero_for_equal_tensors(prediction: torch.Tensor, device: str) -> None:
    loss = GMSDLoss()
    target = prediction.clone()
    measure = loss(prediction.to(device), target.to(device))
    assert measure.abs() <= 1e-6, f'GMSD for equal tensors must be 0, got {measure}'


def test_gmsd_loss_raises_if_tensors_have_different_types(target: torch.Tensor, device: str) -> None:
    wrong_type_predictions = [list(range(10)), np.arange(10)]
    for wrong_type_prediction in wrong_type_predictions:
        with pytest.raises(AssertionError):
            GMSDLoss()(wrong_type_prediction, target.to(device))


@pytest.mark.parametrize(
    "data_range", [128, 255],
)
def test_gmsd_loss_supports_different_data_ranges(
        prediction: torch.Tensor, target: torch.Tensor, data_range, device: str) -> None:

    prediction_scaled = (prediction * data_range).type(torch.uint8)
    target_scaled = (target * data_range).type(torch.uint8)
    loss_scaled = GMSDLoss(data_range=data_range)
    measure_scaled = loss_scaled(prediction_scaled.to(device), target_scaled.to(device))

    loss = GMSDLoss()
    measure = loss(
        prediction_scaled.to(device) / float(data_range),
        target_scaled.to(device) / float(data_range),
    )
    diff = torch.abs(measure_scaled - measure)
    assert diff <= 1e-6, f'Result for same tensor with different data_range should be the same, got {diff}'


def test_gmsd_loss_supports_greyscale_tensors(device: str) -> None:
    loss = GMSDLoss()
    target = torch.ones(2, 1, 96, 96)
    prediction = torch.zeros(2, 1, 96, 96)
    loss(prediction.to(device), target.to(device))


def test_gmsd_loss_modes(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    for reduction in ['mean', 'sum', 'none']:
        GMSDLoss(reduction=reduction)(prediction.to(device), target.to(device))

    for reduction in ['DEADBEEF', 'random']:
        with pytest.raises(KeyError):
            GMSDLoss(reduction=reduction)(prediction.to(device), target.to(device))


# ================== Test function: `multi_scale_gmsd` ==================
def test_multi_scale_gmsd_forward_backward(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    multi_scale_gmsd(prediction.to(device), target.to(device), chromatic=True)


def test_multi_scale_gmsd_zero_for_equal_tensors(prediction: torch.Tensor, device: str) -> None:
    target = prediction.clone()
    measure = multi_scale_gmsd(prediction.to(device), target.to(device))
    assert measure.abs() <= 1e-6, f'MultiScaleGMSD for equal tensors must be 0, got {measure}'


@pytest.mark.parametrize(
    "data_range", [128, 255],
)
def test_multi_scale_gmsd_supports_different_data_ranges(
        prediction: torch.Tensor, target: torch.Tensor, data_range, device: str) -> None:
    prediction_scaled = (prediction * data_range).type(torch.uint8)
    target_scaled = (target * data_range).type(torch.uint8)
    measure_scaled = multi_scale_gmsd(prediction_scaled.to(device), target_scaled.to(device), data_range=data_range)
    measure = multi_scale_gmsd(
        prediction_scaled.to(device) / float(data_range),
        target_scaled.to(device) / float(data_range),
        data_range=1.0
    )
    diff = torch.abs(measure_scaled - measure)
    assert diff <= 1e-6, f'Result for same tensor with different data_range should be the same, got {diff}'


def test_multi_scale_gmsd_fails_for_incorrect_data_range(
        prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    # Scale to [0, 255]
    prediction_scaled = (prediction * 255).type(torch.uint8)
    target_scaled = (target * 255).type(torch.uint8)
    with pytest.raises(AssertionError):
        multi_scale_gmsd(prediction_scaled.to(device), target_scaled.to(device), data_range=1.0)


def test_multi_scale_gmsd_supports_greyscale_tensors(device: str) -> None:
    target = torch.ones(2, 1, 96, 96)
    prediction = torch.zeros(2, 1, 96, 96)
    multi_scale_gmsd(prediction.to(device), target.to(device))


def test_multi_scale_gmsd_fails_for_greyscale_tensors_chromatic_flag(device: str) -> None:
    target = torch.ones(2, 1, 96, 96)
    prediction = torch.zeros(2, 1, 96, 96)
    with pytest.raises(AssertionError):
        multi_scale_gmsd(prediction.to(device), target.to(device), chromatic=True)


def test_multi_scale_gmsd_supports_custom_weights(prediction: torch.Tensor, target: torch.Tensor,
                                                  device: str) -> None:
    multi_scale_gmsd(prediction.to(device), target.to(device), scale_weights=[3., 4., 2., 1., 2.])
    multi_scale_gmsd(prediction.to(device), target.to(device), scale_weights=torch.tensor([3., 4., 2., 1., 2.]))


def test_multi_scale_gmsd_raise_exception_for_small_images(device: str) -> None:
    target = torch.ones(3, 1, 32, 32)
    prediction = torch.zeros(3, 1, 32, 32)
    with pytest.raises(ValueError):
        multi_scale_gmsd(prediction.to(device), target.to(device), scale_weights=[3., 4., 2., 1., 1.])


def test_multi_scale_gmsd_modes(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    for reduction in ['mean', 'sum', 'none']:
        multi_scale_gmsd(prediction.to(device), target.to(device), reduction=reduction)

    for reduction in ['DEADBEEF', 'random']:
        with pytest.raises(KeyError):
            multi_scale_gmsd(prediction.to(device), target.to(device), reduction=reduction)


# ================== Test class: `MultiScaleGMSDLoss` ==================
def test_multi_scale_gmsd_loss_forward_backward(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    prediction.requires_grad_()
    loss_value = MultiScaleGMSDLoss(chromatic=True)(prediction.to(device), target.to(device))
    loss_value.backward()
    assert torch.isfinite(prediction.grad).all(), LEAF_VARIABLE_ERROR_MESSAGE
    

def test_multi_scale_gmsd_loss_zero_for_equal_tensors(prediction: torch.Tensor, device: str) -> None:
    loss = MultiScaleGMSDLoss()
    target = prediction.clone()
    measure = loss(prediction.to(device), target.to(device))
    assert measure.abs() <= 1e-6, f'MultiScaleGMSD for equal tensors must be 0, got {measure}'


def test_multi_scale_gmsd_loss_supports_different_data_ranges(prediction: torch.Tensor, target: torch.Tensor,
                                                              device: str) -> None:
    prediction_255 = prediction * 255
    target_255 = target * 255
    loss = MultiScaleGMSDLoss()
    measure = loss(prediction.to(device), target.to(device))
    loss_255 = MultiScaleGMSDLoss(data_range=255)
    measure_255 = loss_255(prediction_255.to(device), target_255.to(device))
    diff = torch.abs(measure_255 - measure)
    assert diff <= 1e-4, f'Result for same tensor with different data_range should be the same, got {diff}'


def test_multi_scale_gmsd_loss_supports_greyscale_tensors(device: str) -> None:
    loss = MultiScaleGMSDLoss()
    target = torch.ones(2, 1, 96, 96)
    prediction = torch.zeros(2, 1, 96, 96)
    loss(prediction.to(device), target.to(device))


def test_multi_scale_gmsd_loss_fails_for_greyscale_tensors_chromatic_flag(device: str) -> None:
    loss = MultiScaleGMSDLoss(chromatic=True)
    target = torch.ones(2, 1, 96, 96)
    prediction = torch.zeros(2, 1, 96, 96)
    with pytest.raises(AssertionError):
        loss(prediction.to(device), target.to(device))


def test_multi_scale_gmsd_loss_supports_custom_weights(prediction: torch.Tensor, target: torch.Tensor,
                                                       device: str) -> None:
    loss = MultiScaleGMSDLoss(scale_weights=[3., 4., 2., 1., 2.])
    loss(prediction.to(device), target.to(device))
    loss = MultiScaleGMSDLoss(scale_weights=torch.tensor([3., 4., 2., 1., 2.]))
    loss(prediction.to(device), target.to(device))


def test_multi_scale_gmsd_loss_raise_exception_for_small_images(device: str) -> None:
    target = torch.ones(3, 1, 32, 32)
    prediction = torch.zeros(3, 1, 32, 32)
    loss = MultiScaleGMSDLoss(scale_weights=[3., 4., 2., 1., 1.])
    with pytest.raises(ValueError):
        loss(prediction.to(device), target.to(device))


def test_multi_scale_loss_gmsd_modes(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    for reduction in ['mean', 'sum', 'none']:
        MultiScaleGMSDLoss(reduction=reduction)(prediction.to(device), target.to(device))

    for reduction in ['DEADBEEF', 'random']:
        with pytest.raises(KeyError):
            MultiScaleGMSDLoss(reduction=reduction)(prediction.to(device), target.to(device))
