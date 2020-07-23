import torch
import pytest

from piq import gmsd, multi_scale_gmsd, GMSDLoss, MultiScaleGMSDLoss

LEAF_VARIABLE_ERROR_MESSAGE = 'Expected non None gradient of leaf variable'


@pytest.fixture(scope='module')
def prediction() -> torch.Tensor:
    return torch.rand(2, 3, 128, 128)


@pytest.fixture(scope='module')
def target() -> torch.Tensor:
    return torch.rand(2, 3, 128, 128)


# ================== Test function: `gmsd` ==================
def test_gmsd_forward(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    gmsd(prediction.to(device), target.to(device))


def test_gmsd_zero_for_equal_tensors(prediction: torch.Tensor, device: str):
    target = prediction.clone()
    measure = gmsd(prediction.to(device), target.to(device))
    assert measure.abs() <= 1e-6, f'GMSD for equal tensors must be 0, got {measure}'


def test_gmsd_raises_if_tensors_have_different_types(target: torch.Tensor, device: str) -> None:
    wrong_type_prediction = list(range(10))
    with pytest.raises(AssertionError):
        gmsd(wrong_type_prediction, target.to(device))


def test_gmsd_supports_different_data_ranges(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    prediction_255 = (prediction * 255).type(torch.uint8)
    target_255 = (target * 255).type(torch.uint8)
    measure = gmsd(prediction.to(device), target.to(device))

    measure_255 = gmsd(prediction_255.to(device), target_255.to(device), data_range=255)
    diff = torch.abs(measure_255 - measure)
    assert diff <= 1e-4, f'Result for same tensor with different data_range should be the same, got {diff}'


def test_gmsd_supports_greyscale_tensors(device: str):
    target = torch.ones(2, 1, 128, 128)
    prediction = torch.zeros(2, 1, 128, 128)
    gmsd(prediction.to(device), target.to(device))


def test_gmsd_modes(prediction: torch.Tensor, target: torch.Tensor, device: str):
    for reduction in ['mean', 'sum', 'none']:
        gmsd(prediction.to(device), target.to(device), reduction=reduction)

    for reduction in ['DEADBEEF', 'random']:
        with pytest.raises(KeyError):
            gmsd(prediction.to(device), target.to(device), reduction=reduction)


# ================== Test class: `GMSDLoss` ==================
def test_gmsd_loss_forward_backward(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    prediction.requires_grad_()
    loss_value = GMSDLoss()(prediction.to(device), target.to(device))
    loss_value.backward()
    assert torch.isfinite(prediction.grad).all(), LEAF_VARIABLE_ERROR_MESSAGE


def test_gmsd_loss_zero_for_equal_tensors(prediction: torch.Tensor, device: str):
    loss = GMSDLoss()
    target = prediction.clone()
    measure = loss(prediction.to(device), target.to(device))
    assert measure.abs() <= 1e-6, f'GMSD for equal tensors must be 0, got {measure}'


def test_gmsd_loss_raises_if_tensors_have_different_types(target: torch.Tensor, device: str) -> None:
    wrong_type_prediction = list(range(10))
    with pytest.raises(AssertionError):
        GMSDLoss()(wrong_type_prediction, target.to(device))


def test_gmsd_loss_supports_different_data_ranges(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    prediction_255 = (prediction * 255).type(torch.uint8)
    target_255 = (target * 255).type(torch.uint8)
    loss = GMSDLoss()
    measure = loss(prediction.to(device), target.to(device))

    loss_255 = GMSDLoss(data_range=255)
    measure_255 = loss_255(prediction_255.to(device), target_255.to(device))
    diff = torch.abs(measure_255 - measure)
    assert diff <= 1e-4, f'Result for same tensor with different data_range should be the same, got {diff}'


def test_gmsd_loss_supports_greyscale_tensors(device: str):
    loss = GMSDLoss()
    target = torch.ones(2, 1, 128, 128)
    prediction = torch.zeros(2, 1, 128, 128)
    loss(prediction.to(device), target.to(device))


def test_gmsd_loss_modes(prediction: torch.Tensor, target: torch.Tensor, device: str):
    for reduction in ['mean', 'sum', 'none']:
        GMSDLoss(reduction=reduction)(prediction.to(device), target.to(device))

    for reduction in ['DEADBEEF', 'random']:
        with pytest.raises(KeyError):
            GMSDLoss(reduction=reduction)(prediction.to(device), target.to(device))


# ================== Test function: `multi_scale_gmsd` ==================
def test_multi_scale_gmsd_forward_backward(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    multi_scale_gmsd(prediction.to(device), target.to(device), chromatic=True)


def test_multi_scale_gmsd_zero_for_equal_tensors(prediction: torch.Tensor, device: str):
    target = prediction.clone()
    measure = multi_scale_gmsd(prediction.to(device), target.to(device))
    assert measure.abs() <= 1e-6, f'MultiScaleGMSD for equal tensors must be 0, got {measure}'


def test_multi_scale_gmsd_supports_different_data_ranges(prediction: torch.Tensor, target: torch.Tensor,
                                                         device: str) -> None:
    prediction_255 = (prediction * 255).type(torch.uint8)
    target_255 = (target * 255).type(torch.uint8)

    measure = multi_scale_gmsd(prediction.to(device), target.to(device))
    measure_255 = multi_scale_gmsd(prediction_255.to(device), target_255.to(device), data_range=255)
    diff = torch.abs(measure_255 - measure)
    assert diff <= 1e-4, f'Result for same tensor with different data_range should be the same, got {diff}'


def test_multi_scale_gmsd_supports_greyscale_tensors(device: str):
    target = torch.ones(2, 1, 128, 128)
    prediction = torch.zeros(2, 1, 128, 128)
    multi_scale_gmsd(prediction.to(device), target.to(device))


def test_multi_scale_gmsd_fails_for_greyscale_tensors_chromatic_flag(device: str):
    target = torch.ones(2, 1, 128, 128)
    prediction = torch.zeros(2, 1, 128, 128)
    with pytest.raises(AssertionError):
        multi_scale_gmsd(prediction.to(device), target.to(device), chromatic=True)


def test_multi_scale_gmsd_supports_custom_weights(prediction: torch.Tensor, target: torch.Tensor, device: str):
    multi_scale_gmsd(prediction.to(device), target.to(device), scale_weights=[3., 4., 2., 1., 2.])
    multi_scale_gmsd(prediction.to(device), target.to(device), scale_weights=torch.tensor([3., 4., 2., 1., 2.]))


def test_multi_scale_gmsd_raise_exception_for_small_images(device: str):
    target = torch.ones(3, 1, 32, 32)
    prediction = torch.zeros(3, 1, 32, 32)
    with pytest.raises(ValueError):
        multi_scale_gmsd(prediction.to(device), target.to(device), scale_weights=[3., 4., 2., 1., 1.])


def test_multi_scale_gmsd_modes(prediction: torch.Tensor, target: torch.Tensor, device: str):
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
    

def test_multi_scale_gmsd_loss_zero_for_equal_tensors(prediction: torch.Tensor, device: str):
    loss = MultiScaleGMSDLoss()
    target = prediction.clone()
    measure = loss(prediction.to(device), target.to(device))
    assert measure.abs() <= 1e-6, f'MultiScaleGMSD for equal tensors must be 0, got {measure}'


def test_multi_scale_gmsd_loss_supports_different_data_ranges(prediction: torch.Tensor, target: torch.Tensor,
                                                              device: str) -> None:
    prediction_255 = (prediction * 255).type(torch.uint8)
    target_255 = (target * 255).type(torch.uint8)
    loss = MultiScaleGMSDLoss()
    measure = loss(prediction.to(device), target.to(device))
    loss_255 = MultiScaleGMSDLoss(data_range=255)
    measure_255 = loss_255(prediction_255.to(device), target_255.to(device))
    diff = torch.abs(measure_255 - measure)
    assert diff <= 1e-4, f'Result for same tensor with different data_range should be the same, got {diff}'


def test_multi_scale_gmsd_loss_supports_greyscale_tensors(device: str):
    loss = MultiScaleGMSDLoss()
    target = torch.ones(2, 1, 128, 128)
    prediction = torch.zeros(2, 1, 128, 128)
    loss(prediction.to(device), target.to(device))


def test_multi_scale_gmsd_loss_fails_for_greyscale_tensors_chromatic_flag(device: str):
    loss = MultiScaleGMSDLoss(chromatic=True)
    target = torch.ones(2, 1, 128, 128)
    prediction = torch.zeros(2, 1, 128, 128)
    with pytest.raises(AssertionError):
        loss(prediction.to(device), target.to(device))


def test_multi_scale_gmsd_loss_supports_custom_weights(prediction: torch.Tensor, target: torch.Tensor, device: str):
    loss = MultiScaleGMSDLoss(scale_weights=[3., 4., 2., 1., 2.])
    loss(prediction.to(device), target.to(device))
    loss = MultiScaleGMSDLoss(scale_weights=torch.tensor([3., 4., 2., 1., 2.]))
    loss(prediction.to(device), target.to(device))


def test_multi_scale_gmsd_loss_raise_exception_for_small_images(device: str):
    target = torch.ones(3, 1, 32, 32)
    prediction = torch.zeros(3, 1, 32, 32)
    loss = MultiScaleGMSDLoss(scale_weights=[3., 4., 2., 1., 1.])
    with pytest.raises(ValueError):
        loss(prediction.to(device), target.to(device))


def test_multi_scale_loss_gmsd_modes(prediction: torch.Tensor, target: torch.Tensor, device: str):
    for reduction in ['mean', 'sum', 'none']:
        MultiScaleGMSDLoss(reduction=reduction)(prediction.to(device), target.to(device))

    for reduction in ['DEADBEEF', 'random']:
        with pytest.raises(KeyError):
            MultiScaleGMSDLoss(reduction=reduction)(prediction.to(device), target.to(device))
