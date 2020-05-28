import torch
import pytest

from photosynthesis_metrics import GMSDLoss, MultiScaleGMSDLoss


@pytest.fixture(scope='module')
def prediction() -> torch.Tensor:
    return torch.rand(3, 3, 256, 256)


@pytest.fixture(scope='module')
def target() -> torch.Tensor:
    return torch.rand(3, 3, 256, 256)


# ================== Test class: `GMSDLoss` ==================
def test_gmsd_loss_init() -> None:
    try:
        GMSDLoss()
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_gmsd_zero_for_equal_tensors(prediction: torch.Tensor):
    loss = GMSDLoss()
    target = prediction.clone()
    measure = loss(prediction, target)
    assert measure.abs() <= 1e-6, f'GMSD for equal tensors must be 0, got {measure}'


def test_gmsd_loss_raises_if_tensors_have_different_types(target: torch.Tensor) -> None:
    wrong_type_prediction = list(range(10))
    with pytest.raises(AssertionError):
        GMSDLoss()(wrong_type_prediction, target)


def test_gmsd_loss_supports_different_data_ranges(prediction: torch.Tensor, target: torch.Tensor) -> None:
    prediction_255 = (prediction * 255).type(torch.uint8)
    target_255 = (target * 255).type(torch.uint8)
    loss = GMSDLoss()
    measure = loss(prediction, target)

    loss_255 = GMSDLoss(data_range=255)
    measure_255 = loss_255(prediction_255, target_255)
    diff = torch.abs(measure_255 - measure)
    assert diff <= 1e-4, f'Result for same tensor with different data_range should be the same, got {diff}'


def test_gmsd_supports_greyscale_tensors():
    loss = GMSDLoss()
    target = torch.ones(3, 1, 256, 256)
    prediction = torch.zeros(3, 1, 256, 256)
    try:
        loss(prediction, target)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


# ================== Test class: `MultiScaleGMSDLoss` ==================
def test_multi_scale_gmsd_loss_init() -> None:
    try:
        MultiScaleGMSDLoss()
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_multi_scale_gmsd_zero_for_equal_tensors(prediction: torch.Tensor):
    loss = MultiScaleGMSDLoss()
    target = prediction.clone()
    measure = loss(prediction, target)
    assert measure.abs() <= 1e-6, f'MultiScaleGMSD for equal tensors must be 0, got {measure}'


def test_multi_scale_gmsd_loss_supports_different_data_ranges(
        prediction: torch.Tensor, target: torch.Tensor) -> None:
    prediction_255 = (prediction * 255).type(torch.uint8)
    target_255 = (target * 255).type(torch.uint8)
    loss = MultiScaleGMSDLoss()
    measure = loss(prediction, target)
    loss_255 = MultiScaleGMSDLoss(data_range=255)
    measure_255 = loss_255(prediction_255, target_255)
    diff = torch.abs(measure_255 - measure)
    assert diff <= 1e-4, f'Result for same tensor with different data_range should be the same, got {diff}'


def test_multi_scale_gmsd_supports_greyscale_tensors():
    loss = MultiScaleGMSDLoss()
    target = torch.ones(3, 1, 256, 256)
    prediction = torch.zeros(3, 1, 256, 256)
    try:
        loss(prediction, target)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_multi_scale_gmsd_fails_for_greyscale_tensors_chromatic_flag():
    loss = MultiScaleGMSDLoss(chromatic=True)
    target = torch.ones(3, 1, 256, 256)
    prediction = torch.zeros(3, 1, 256, 256)
    with pytest.raises(AssertionError):
        loss(prediction, target)


def test_multi_scale_gmsd_supports_custom_scale_weights(prediction: torch.Tensor, target: torch.Tensor):
    try:
        loss = MultiScaleGMSDLoss(scale_weights=[3., 4., 2., 1., 2.])
        loss(prediction, target)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_multi_scale_gmsd_raise_exception_for_small_images():
    target = torch.ones(3, 1, 32, 32)
    prediction = torch.zeros(3, 1, 32, 32)
    loss = MultiScaleGMSDLoss(scale_weights=[3., 4., 2., 1., 1.])
    with pytest.raises(ValueError):
        loss(prediction, target)
