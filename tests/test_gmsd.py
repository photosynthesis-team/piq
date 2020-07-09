import torch
import pytest

from piq import GMSDLoss, MultiScaleGMSDLoss

LEAF_VARIABLE_ERROR_MESSAGE = 'Expected non None gradient of leaf variable'


@pytest.fixture(scope='module')
def prediction() -> torch.Tensor:
    return torch.rand(2, 3, 128, 128)


@pytest.fixture(scope='module')
def target() -> torch.Tensor:
    return torch.rand(2, 3, 128, 128)


# ================== Test class: `GMSDLoss` ==================
def test_gmsd_loss(prediction: torch.Tensor, target: torch.Tensor) -> None:
    loss = GMSDLoss()
    loss(prediction, target)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test if there is no GPU.')
def test_gmsd_loss_on_gpu(prediction: torch.Tensor, target: torch.Tensor) -> None:
    loss = GMSDLoss()
    loss(prediction.cuda(), target.cuda())


def test_gmsd_loss_backward(prediction: torch.Tensor, target: torch.Tensor) -> None:
    prediction.requires_grad_()
    loss_value = GMSDLoss()(prediction, target)
    loss_value.backward()
    assert prediction.grad is not None, LEAF_VARIABLE_ERROR_MESSAGE


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test if there is no GPU.')
def test_gmsd_loss_backward_on_gpu(prediction: torch.Tensor, target: torch.Tensor) -> None:
    prediction.requires_grad_()
    loss_value = GMSDLoss()(prediction.cuda(), target.cuda())
    loss_value.backward()
    assert prediction.grad is not None, LEAF_VARIABLE_ERROR_MESSAGE


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
    target = torch.ones(2, 1, 128, 128)
    prediction = torch.zeros(2, 1, 128, 128)
    loss(prediction, target)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test if there is no GPU.')
def test_gmsd_supports_greyscale_tensors_on_gpu():
    loss = GMSDLoss()
    target = torch.ones(2, 1, 128, 128).cuda()
    prediction = torch.zeros(2, 1, 128, 128).cuda()
    loss(prediction, target)


# ================== Test class: `MultiScaleGMSDLoss` ==================
def test_multi_scale_gmsd_loss(prediction: torch.Tensor, target: torch.Tensor) -> None:
    loss = MultiScaleGMSDLoss(chromatic=True)
    loss(prediction, target)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test if there is no GPU.')
def test_multi_scale_gmsd_loss_on_gpu(prediction: torch.Tensor, target: torch.Tensor) -> None:
    prediction = prediction.cuda()
    target = target.cuda()
    loss = MultiScaleGMSDLoss(chromatic=True)
    loss(prediction, target)


def test_multi_scale_gmsd_loss_backward(prediction: torch.Tensor, target: torch.Tensor) -> None:
    prediction.requires_grad_()
    loss_value = MultiScaleGMSDLoss(chromatic=True)(prediction, target)
    loss_value.backward()
    assert prediction.grad is not None, LEAF_VARIABLE_ERROR_MESSAGE


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test if there is no GPU.')
def test_multi_scale_gmsd_loss_backward_on_gpu(prediction: torch.Tensor, target: torch.Tensor) -> None:
    prediction.requires_grad_()
    loss_value = MultiScaleGMSDLoss(chromatic=True)(prediction, target)
    loss_value.backward()
    assert prediction.grad is not None, LEAF_VARIABLE_ERROR_MESSAGE
    

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
    target = torch.ones(2, 1, 128, 128)
    prediction = torch.zeros(2, 1, 128, 128)
    loss(prediction, target)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test if there is no GPU.')
def test_multi_scale_gmsd_supports_greyscale_tensors_on_gpu():
    loss = MultiScaleGMSDLoss()
    target = torch.ones(2, 1, 128, 128).cuda()
    prediction = torch.zeros(2, 1, 128, 128).cuda()
    loss(prediction.cuda(), target.cuda())


def test_multi_scale_gmsd_fails_for_greyscale_tensors_chromatic_flag():
    loss = MultiScaleGMSDLoss(chromatic=True)
    target = torch.ones(2, 1, 128, 128)
    prediction = torch.zeros(2, 1, 128, 128)
    with pytest.raises(AssertionError):
        loss(prediction, target)


def test_multi_scale_gmsd_supports_custom_weights(
        prediction: torch.Tensor, target: torch.Tensor):
    loss = MultiScaleGMSDLoss(scale_weights=[3., 4., 2., 1., 2.])
    loss(prediction, target)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test if there is no GPU.')
def test_multi_scale_gmsd_supports_custom_weights_on_gpu(
        prediction: torch.Tensor, target: torch.Tensor):
    loss = MultiScaleGMSDLoss(scale_weights=[3., 4., 2., 1., 2.])
    loss(prediction, target)


def test_multi_scale_gmsd_raise_exception_for_small_images():
    target = torch.ones(3, 1, 32, 32)
    prediction = torch.zeros(3, 1, 32, 32)
    loss = MultiScaleGMSDLoss(scale_weights=[3., 4., 2., 1., 1.])
    with pytest.raises(ValueError):
        loss(prediction, target)
