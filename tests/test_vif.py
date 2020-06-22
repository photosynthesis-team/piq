import torch
import pytest

from piq import VIFLoss, vif_p


@pytest.fixture(scope='module')
def prediction() -> torch.Tensor:
    return torch.rand(4, 3, 256, 256)


@pytest.fixture(scope='module')
def target() -> torch.Tensor:
    return torch.rand(4, 3, 256, 256)


@pytest.fixture(scope='module')
def prediction_1d() -> torch.Tensor:
    return torch.rand(4, 1, 256, 256)


@pytest.fixture(scope='module')
def target_1d() -> torch.Tensor:
    return torch.rand(4, 1, 256, 256)


# ================== Test function: `vif_p` ==================
def test_vif_p_works_for_3_channels(prediction: torch.Tensor, target: torch.Tensor) -> None:
    vif_p(prediction, target, data_range=1.)


def test_vif_p_works_for_1_channel(prediction_1d: torch.Tensor, target_1d: torch.Tensor) -> None:
    vif_p(prediction_1d, target_1d, data_range=1.)


def test_vif_p_one_for_equal_tensors(prediction: torch.Tensor) -> None:
    target = prediction.clone()
    measure = vif_p(prediction, target)
    assert torch.isclose(measure, torch.tensor(1.0)), f'VIF for equal tensors shouls be 1.0, got {measure}.'


def test_vif_p_works_for_zeros_tensors() -> None:
    prediction = torch.zeros(4, 3, 256, 256)
    target = torch.zeros(4, 3, 256, 256)
    measure = vif_p(prediction, target, data_range=1.)
    assert torch.isclose(measure, torch.tensor(1.0)), f'VIF for 2 zero tensors shouls be 1.0, got {measure}.'


def test_vif_p_works_for_different_data_range(prediction: torch.Tensor, target: torch.Tensor) -> None:
    prediction_255 = (prediction * 255).type(torch.uint8)
    target_255 = (target * 255).type(torch.uint8)
    vif_p(prediction_255, target_255, data_range=255)


def test_vif_p_fails_for_small_images() -> None:
    prediction = torch.rand(2, 3, 32, 32)
    target = torch.rand(2, 3, 32, 32)
    with pytest.raises(ValueError):
        vif_p(prediction, target)


# ================== Test class: `VIFLoss` ==================
def test_vif_loss_forward(prediction: torch.Tensor, target: torch.Tensor) -> None:
    loss = VIFLoss()
    loss(prediction, target)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test if there is no GPU.')
def test_vif_loss_forward_on_gpu(prediction: torch.Tensor, target: torch.Tensor) -> None:
    loss = VIFLoss()
    loss(prediction.cuda(), target.cuda())


def test_vif_loss_zero_for_equal_tensors(prediction: torch.Tensor):
    loss = VIFLoss()
    target = prediction.clone()
    measure = loss(prediction, target)
    assert torch.isclose(measure, torch.tensor(0.), atol=1e-6), f'VIF for equal tensors must be 0, got {measure}'


def test_vif_loss_reduction(prediction: torch.Tensor, target: torch.Tensor) -> None:
    loss = VIFLoss(reduction='mean')
    measure = loss(prediction, target)
    assert measure.dim() == 0, f'VIF with `mean` reduction must return 1 number, got {len(measure)}'

    loss = VIFLoss(reduction='sum')
    measure = loss(prediction, target)
    assert measure.dim() == 0, f'VIF with `mean` reduction must return 1 number, got {len(measure)}'

    loss = VIFLoss(reduction='none')
    measure = loss(prediction, target)
    assert len(measure) == prediction.size(0), \
        f'VIF with `none` reduction must have length equal to number of images, got {len(measure)}'
    
    loss = VIFLoss(reduction='random string')
    with pytest.raises(KeyError):
        loss(prediction, target)


NONE_GRAD_ERR_MSG = 'Expected non None gradient of leaf variable'


def test_vif_loss_computes_grad(prediction: torch.Tensor, target: torch.Tensor) -> None:
    prediction.requires_grad_()
    loss_value = VIFLoss()(prediction, target)
    loss_value.backward()
    assert prediction.grad is not None, NONE_GRAD_ERR_MSG


def test_vif_loss_computes_grad_for_zeros_tensors() -> None:
    prediction = torch.zeros(4, 3, 256, 256, requires_grad=True)
    target = torch.zeros(4, 3, 256, 256)
    loss_value = VIFLoss()(prediction, target)
    loss_value.backward()
    assert prediction.grad is not None, NONE_GRAD_ERR_MSG


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test if there is no GPU.')
def test_vif_loss_computes_grad_on_gpu(prediction: torch.Tensor, target: torch.Tensor) -> None:
    prediction.requires_grad_()
    loss_value = VIFLoss()(prediction.cuda(), target.cuda())
    loss_value.backward()
    assert prediction.grad is not None, NONE_GRAD_ERR_MSG
