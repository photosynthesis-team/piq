import torch
from piq import vsi, VSILoss
from PIL import Image
import numpy as np


# ================== Test function: `vsi` ==================
def test_vsi_to_be_one_for_identical_inputs(input_tensors, device) -> None:
    prediction, _ = input_tensors
    index = vsi(prediction.to(device), prediction.to(device), data_range=1., reduction='none')
    index_255 = vsi(prediction.to(device) * 255, prediction.to(device) * 255, data_range=255, reduction='none')
    assert torch.isclose(index, torch.ones_like(index, device=device)).all(), \
        f'Expected index to be equal 1, got {index}'
    assert torch.isclose(index_255, torch.ones_like(index_255, device=device)).all(), \
        f'Expected index to be equal 1, got {index_255}'


def test_vsi_symmetry(input_tensors, device) -> None:
    prediction, target = input_tensors
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


def test_vsi_compare_with_matlab(device) -> None:
    prediction = torch.tensor(np.array(Image.open('tests/assets/I01.BMP'))).permute(2, 0, 1)
    target = torch.tensor(np.array(Image.open('tests/assets/i01_01_5.bmp'))).permute(2, 0, 1)
    predicted_score = vsi(prediction, target, data_range=255, reduction='none')
    target_score = torch.tensor([0.96405]).to(predicted_score)
    assert torch.allclose(predicted_score, target_score), f'Expected result similar to MATLAB,' \
                                                          f'got diff{predicted_score - target_score}'


# ================== Test class: `VSILoss` =================
def test_vsi_loss(input_tensors, device) -> None:
    prediction, target = input_tensors
    prediction.requires_grad_()
    loss = VSILoss(data_range=1.)(prediction.to(device), target.to(device))
    loss.backward()
    assert prediction.grad is not None, f'Expected finite gradient values after back propagation, got {prediction.grad}'


def test_vsi_loss_zero_for_equal_input(input_tensors, device) -> None:
    prediction, _ = input_tensors
    target = prediction.clone()
    prediction.requires_grad_()
    loss = VSILoss(data_range=1.)(prediction.to(device), target.to(device))
    assert torch.isclose(loss, torch.zeros_like(loss)), \
        f'Expected loss equals zero for identical inputs, got {loss}'
