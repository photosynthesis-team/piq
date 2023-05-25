import torch
import pytest
from skimage.io import imread
import numpy as np
from typing import Any, Tuple

from piq import gmsd, multi_scale_gmsd, GMSDLoss, MultiScaleGMSDLoss

LEAF_VARIABLE_ERROR_MESSAGE = 'Expected non None gradient of leaf variable'


@pytest.fixture(scope='module')
def x() -> torch.Tensor:
    return torch.rand(2, 3, 96, 96)


@pytest.fixture(scope='module')
def y() -> torch.Tensor:
    return torch.rand(2, 3, 96, 96)


x_image = [
    torch.tensor(imread('tests/assets/goldhill_jpeg.gif'), dtype=torch.float32).unsqueeze(0).unsqueeze(0),
    torch.tensor(imread('tests/assets/i01_01_5.bmp'), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
]

y_image = [
    torch.tensor(imread('tests/assets/goldhill.gif'), dtype=torch.float32).unsqueeze(0).unsqueeze(0),
    torch.tensor(imread('tests/assets/I01.BMP'), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
]

y_score = [
    torch.tensor(0.138012587141798),
    torch.tensor(0.094124655829098)
]


@pytest.fixture(params=zip(x_image, y_image, y_score))
def input_images_score(request: Any) -> Any:
    return request.param


# ================== Test function: `gmsd` ==================
def test_gmsd_forward(x, y, device: str) -> None:
    gmsd(x.to(device), y.to(device))


def test_gmsd_zero_for_equal_tensors(x, device: str) -> None:
    y = x.clone()
    measure = gmsd(x.to(device), y.to(device))
    assert measure.abs() <= 1e-6, f'GMSD for equal tensors must be 0, got {measure}'


def test_gmsd_raises_if_tensors_have_different_types(y, device: str) -> None:
    wrong_type_x = [list(range(10)), np.arange(10)]
    for wrong_type_x in wrong_type_x:
        with pytest.raises(AssertionError):
            gmsd(wrong_type_x, y.to(device))


@pytest.mark.parametrize(
    "data_range", [128, 255],
)
def test_gmsd_supports_different_data_ranges(x, y, data_range, device: str) -> None:
    x_scaled = (x * data_range).to(dtype=torch.uint8, device=device)
    y_scaled = (y * data_range).to(dtype=torch.uint8, device=device)
    measure_scaled = gmsd(x_scaled, y_scaled, data_range=data_range)
    measure = gmsd(
        x_scaled / float(data_range),
        y_scaled / float(data_range),
        data_range=1.0
    )
    diff = torch.abs(measure_scaled - measure)
    assert diff <= 1e-6, f'Result for same tensor with different data_range should be the same, got {diff}'


def test_gmsd_fails_for_incorrect_data_range(x, y, device: str) -> None:
    # Scale to [0, 255]
    x_scaled = (x * 255).to(dtype=torch.uint8, device=device)
    y_scaled = (y * 255).to(dtype=torch.uint8, device=device)
    with pytest.raises(AssertionError):
        gmsd(x_scaled, y_scaled, data_range=1.0)


def test_gmsd_supports_greyscale_tensors(device: str) -> None:
    y = torch.ones(2, 1, 96, 96, device=device)
    x = torch.zeros(2, 1, 96, 96, device=device)
    gmsd(x, y)


def test_gmsd_modes(x, y, device: str) -> None:
    for reduction in ['mean', 'sum', 'none']:
        gmsd(x.to(device), y.to(device), reduction=reduction)

    for reduction in ['DEADBEEF', 'random']:
        with pytest.raises(ValueError):
            gmsd(x.to(device), y.to(device), reduction=reduction)


def test_gmsd_compare_with_matlab(input_images_score: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                  device: str) -> None:
    x, y, y_value = input_images_score
    score = gmsd(x=x.to(device), y=y.to(device), data_range=255)
    assert torch.isclose(score, y_value.to(score)), f'The estimated value must be equal to MATLAB provided one, ' \
                                                    f'got {score.item():.8f}, while MATLAB equals {y_value}'


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64],
)
def test_gmsd_preserves_dtype(x, y, dtype, device: str) -> None:
    output = gmsd(x.to(device=device, dtype=dtype), y.to(device=device, dtype=dtype))
    assert output.dtype == dtype


# ================== Test class: `GMSDLoss` ==================
def test_gmsd_loss_forward_backward(x, y, device: str) -> None:
    x.requires_grad_()
    loss_value = GMSDLoss()(x.to(device), y.to(device))
    loss_value.backward()
    assert torch.isfinite(x.grad).all(), LEAF_VARIABLE_ERROR_MESSAGE


def test_gmsd_loss_zero_for_equal_tensors(x, device: str) -> None:
    loss = GMSDLoss()
    y = x.clone()
    measure = loss(x.to(device), y.to(device))
    assert measure.abs() <= 1e-6, f'GMSD for equal tensors must be 0, got {measure}'


def test_gmsd_loss_raises_if_tensors_have_different_types(y, device: str) -> None:
    wrong_type_x = [list(range(10)), np.arange(10)]
    for wrong_x in wrong_type_x:
        with pytest.raises(AssertionError):
            GMSDLoss()(wrong_x, y.to(device))


@pytest.mark.parametrize(
    "data_range", [128, 255],
)
def test_gmsd_loss_supports_different_data_ranges(x, y, data_range, device: str) -> None:
    x_scaled = (x * data_range).to(dtype=torch.uint8, device=device)
    y_scaled = (y * data_range).to(dtype=torch.uint8, device=device)
    loss_scaled = GMSDLoss(data_range=data_range)
    measure_scaled = loss_scaled(x_scaled, y_scaled)

    loss = GMSDLoss()
    measure = loss(
        x_scaled / float(data_range),
        y_scaled / float(data_range),
    )
    diff = torch.abs(measure_scaled - measure)
    assert diff <= 1e-6, f'Result for same tensor with different data_range should be the same, got {diff}'


def test_gmsd_loss_supports_greyscale_tensors(device: str) -> None:
    loss = GMSDLoss()
    y = torch.ones(2, 1, 96, 96, device=device)
    x = torch.zeros(2, 1, 96, 96, device=device)
    loss(x, y)


def test_gmsd_loss_modes(x, y, device: str) -> None:
    for reduction in ['mean', 'sum', 'none']:
        GMSDLoss(reduction=reduction)(x.to(device), y.to(device))

    for reduction in ['DEADBEEF', 'random']:
        with pytest.raises(ValueError):
            GMSDLoss(reduction=reduction)(x.to(device), y.to(device))


# ================== Test function: `multi_scale_gmsd` ==================
def test_multi_scale_gmsd_forward_backward(x, y, device: str) -> None:
    multi_scale_gmsd(x.to(device), y.to(device), chromatic=True)


def test_multi_scale_gmsd_zero_for_equal_tensors(x, device: str) -> None:
    y = x.clone()
    measure = multi_scale_gmsd(x.to(device), y.to(device))
    assert measure.abs() <= 1e-6, f'MultiScaleGMSD for equal tensors must be 0, got {measure}'


@pytest.mark.parametrize(
    "data_range", [128, 255],
)
def test_multi_scale_gmsd_supports_different_data_ranges(x, y, data_range, device: str) -> None:
    x_scaled = (x * data_range).to(dtype=torch.uint8, device=device)
    y_scaled = (y * data_range).to(dtype=torch.uint8, device=device)
    measure_scaled = multi_scale_gmsd(x_scaled, y_scaled, data_range=data_range)
    measure = multi_scale_gmsd(
        x_scaled / float(data_range),
        y_scaled / float(data_range),
        data_range=1.0
    )
    diff = torch.abs(measure_scaled - measure)
    assert diff <= 1e-6, f'Result for same tensor with different data_range should be the same, got {diff}'


def test_multi_scale_gmsd_fails_for_incorrect_data_range(x, y, device: str) -> None:
    # Scale to [0, 255]
    x_scaled = (x * 255).to(dtype=torch.uint8, device=device)
    y_scaled = (y * 255).to(dtype=torch.uint8, device=device)
    with pytest.raises(AssertionError):
        multi_scale_gmsd(x_scaled, y_scaled, data_range=1.0)


def test_multi_scale_gmsd_supports_greyscale_tensors(device: str) -> None:
    y = torch.ones(2, 1, 96, 96, device=device)
    x = torch.zeros(2, 1, 96, 96, device=device)
    multi_scale_gmsd(x, y)


def test_multi_scale_gmsd_fails_for_greyscale_tensors_chromatic_flag(device: str) -> None:
    y = torch.ones(2, 1, 96, 96, device=device)
    x = torch.zeros(2, 1, 96, 96, device=device)
    with pytest.raises(AssertionError):
        multi_scale_gmsd(x, y, chromatic=True)


def test_multi_scale_gmsd_supports_custom_weights(x, y, device: str) -> None:
    scale_weights = torch.tensor([3., 4., 2., 1., 2.], device=device)
    multi_scale_gmsd(x.to(device), y.to(device), scale_weights=scale_weights)


def test_multi_scale_gmsd_raise_exception_for_small_images(device: str) -> None:
    y = torch.ones(3, 1, 32, 32, device=device)
    x = torch.zeros(3, 1, 32, 32, device=device)
    scale_weights = torch.tensor([3., 4., 2., 1., 2.], device=device)
    with pytest.raises(ValueError):
        multi_scale_gmsd(x, y, scale_weights=scale_weights)


def test_multi_scale_gmsd_modes(x, y, device: str) -> None:
    for reduction in ['mean', 'sum', 'none']:
        multi_scale_gmsd(x.to(device), y.to(device), reduction=reduction)

    for reduction in ['DEADBEEF', 'random']:
        with pytest.raises(ValueError):
            multi_scale_gmsd(x.to(device), y.to(device), reduction=reduction)


# ================== Test class: `MultiScaleGMSDLoss` ==================
def test_multi_scale_gmsd_loss_forward_backward(x, y, device: str) -> None:
    x.requires_grad_()
    loss_value = MultiScaleGMSDLoss(chromatic=True).to(device)(x.to(device), y.to(device))
    loss_value.backward()
    assert torch.isfinite(x.grad).all(), LEAF_VARIABLE_ERROR_MESSAGE


def test_multi_scale_gmsd_loss_zero_for_equal_tensors(x, device: str) -> None:
    loss = MultiScaleGMSDLoss().to(device)
    y = x.clone()
    measure = loss(x.to(device), y.to(device))
    assert measure.abs() <= 1e-6, f'MultiScaleGMSD for equal tensors must be 0, got {measure}'


@pytest.mark.parametrize(
    "data_range", [128, 255],
)
def test_multi_scale_gmsd_loss_supports_different_data_ranges(x, y, data_range, device: str) -> None:
    x_scaled = (x * data_range).to(dtype=torch.uint8, device=device)
    y_scaled = (y * data_range).to(dtype=torch.uint8, device=device)
    loss_scaled = MultiScaleGMSDLoss(data_range=data_range).to(device)
    measure_scaled = loss_scaled(x_scaled, y_scaled)

    loss = MultiScaleGMSDLoss(data_range=1.).to(device)
    measure = loss(x_scaled / float(data_range), y_scaled / float(data_range))

    diff = torch.abs(measure_scaled - measure)
    assert diff <= 1e-6, f'Result for same tensor with different data_range should be the same, got {diff}'


def test_multi_scale_gmsd_loss_supports_greyscale_tensors(device: str) -> None:
    loss = MultiScaleGMSDLoss().to(device)
    y = torch.ones(2, 1, 96, 96, device=device)
    x = torch.zeros(2, 1, 96, 96, device=device)
    loss(x, y)


def test_multi_scale_gmsd_loss_fails_for_greyscale_tensors_chromatic_flag(device: str) -> None:
    loss = MultiScaleGMSDLoss(chromatic=True).to(device)
    y = torch.ones(2, 1, 96, 96, device=device)
    x = torch.zeros(2, 1, 96, 96, device=device)
    with pytest.raises(AssertionError):
        loss(x, y)


def test_multi_scale_gmsd_loss_supports_custom_weights(x, y, device: str) -> None:
    loss = MultiScaleGMSDLoss(scale_weights=torch.tensor([3., 4., 2., 1., 2.])).to(device)
    loss(x.to(device), y.to(device))


def test_multi_scale_gmsd_loss_raise_exception_for_small_images(device: str) -> None:
    y = torch.ones(3, 1, 32, 32, device=device)
    x = torch.zeros(3, 1, 32, 32, device=device)
    loss = MultiScaleGMSDLoss(scale_weights=torch.tensor([3., 4., 2., 1., 2.])).to(device)
    with pytest.raises(ValueError):
        loss(x, y)


def test_multi_scale_loss_gmsd_modes(x, y, device: str) -> None:
    for reduction in ['mean', 'sum', 'none']:
        loss = MultiScaleGMSDLoss(reduction=reduction).to(device)
        loss(x.to(device), y.to(device))

    for reduction in ['DEADBEEF', 'random']:
        with pytest.raises(ValueError):
            loss = MultiScaleGMSDLoss(reduction=reduction).to(device)
            loss(x.to(device), y.to(device))
