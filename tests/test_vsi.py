import torch
import pytest
from piq import vsi, VSILoss


# ================== Test function: `vsi` ==================
def test_vsi_raises_if_one_channel_input() -> None:
    one_channel_prediction = torch.rand(256, 256)
    with pytest.raises(AssertionError):
        vsi(one_channel_prediction, one_channel_prediction, data_range=1.)


def test_vsi_to_be_one_for_identical_inputs(prediction, device) -> None:
    index = vsi(prediction.to(device), prediction.to(device), data_range=1., reduction='none')
    index_255 = vsi(prediction.to(device) * 255, prediction.to(device) * 255, data_range=255, reduction='none')
    assert torch.isclose(index, torch.ones_like(index, device=device)).all(), \
        f'Expected index to be equal 1, got {index}'
    assert torch.isclose(index_255, torch.ones_like(index_255, device=device)).all(), \
        f'Expected index to be equal 1, got {index_255}'


def test_vsi_symmetry(prediction, target, device) -> None:
    result = vsi(prediction.to(device), target.to(device), data_range=1., reduction='none')
    result_sym = vsi(target.to(device), prediction.to(device), data_range=1., reduction='none')
    assert torch.isclose(result_sym, result).all(), f'Expected the same results, got {result} and {result_sym}'


def test_vsi_zeros_ones_inputs(device) -> None:
    zeros = torch.zeros(1, 3, 256, 256, device=device)
    ones = torch.zeros(1, 3, 256, 256, device=device)
    vsi_zeros = vsi(zeros, zeros, data_range=1.)
    assert torch.isfinite(vsi_zeros).all(), f'Expected finite value for zeros tensors, got {vsi_zeros}'
    vsi_ones = vsi(ones, ones, data_range=1.)
    assert torch.isfinite(vsi_ones).all(), f'Expected finite value for ones tensos, got {vsi_ones}'
    vsi_zeros_ones = vsi(zeros, ones, data_range=1.)
    assert torch.isfinite(vsi_zeros_ones).all(), \
        f'Expected finite value for zeros and ones tensos, got {vsi_zeros_ones}'


# ================== Test class: `VSILoss` =================
def test_vsi_loss(prediction, target, device) -> None:
    prediction.requires_grad_()
    loss = VSILoss(data_range=1.)(prediction.to(device), target.to(device))
    loss.backward()
    assert prediction.grad is not None, f'Expected finite gradient values after back propagation, got {prediction.grad}'


def test_vsi_loss_zero_for_equal_input(prediction, device) -> None:
    target = prediction.clone()
    prediction.requires_grad_()
    loss = VSILoss(data_range=1.)(prediction.to(device), target.to(device))
    assert torch.isclose(loss, torch.zeros_like(loss)), \
        f'Expected loss equals zero for identical inputs, got {loss}'
