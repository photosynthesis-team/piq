from typing import Tuple

import torch
import pytest
from piq import vsi, VSILoss
from skimage.io import imread


# ================== Test function: `vsi` ==================
def test_vsi_to_be_one_for_identical_inputs(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    x, _ = input_tensors
    index = vsi(x.to(device), x.to(device), data_range=1., reduction='none')
    index_255 = vsi(x.to(device) * 255, x.to(device) * 255, data_range=255, reduction='none')
    assert torch.allclose(index, torch.ones_like(index, device=device)), \
        f'Expected index to be equal 1, got {index}'
    assert torch.allclose(index_255, torch.ones_like(index_255, device=device)), \
        f'Expected index to be equal 1, got {index_255}'


def test_vsi_symmetry(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    x, y = input_tensors
    result = vsi(x.to(device), y.to(device), data_range=1., reduction='none')
    result_sym = vsi(y.to(device), x.to(device), data_range=1., reduction='none')
    assert torch.allclose(result_sym, result), f'Expected the same results, got {result} and {result_sym}'


def test_vsi_zeros_ones_inputs(device: str) -> None:
    zeros = torch.zeros(1, 3, 128, 128, device=device)
    ones = torch.ones(1, 3, 128, 128, device=device)
    vsi_zeros = vsi(zeros, zeros, data_range=1.)
    assert torch.isfinite(vsi_zeros).all(), f'Expected finite value for zeros tensors, got {vsi_zeros}'
    vsi_ones = vsi(ones, ones, data_range=1.)
    assert torch.isfinite(vsi_ones).all(), f'Expected finite value for ones tensos, got {vsi_ones}'
    vsi_zeros_ones = vsi(zeros, ones, data_range=1.)
    assert torch.isfinite(vsi_zeros_ones).all(), \
        f'Expected finite value for zeros and ones tensos, got {vsi_zeros_ones}'


def test_vsi_compare_with_matlab(device: str) -> None:
    x = torch.tensor(imread('tests/assets/I01.BMP')).permute(2, 0, 1)[None, ...]
    y = torch.tensor(imread('tests/assets/i01_01_5.bmp')).permute(2, 0, 1)[None, ...]
    predicted_score = vsi(x.to(device), y.to(device), data_range=255, reduction='none')
    target_score = torch.tensor([0.96405]).to(predicted_score)
    assert torch.allclose(predicted_score, target_score), f'Expected result similar to MATLAB,' \
                                                          f'got diff{predicted_score - target_score}'


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64],
)
def test_vsi_preserves_dtype(input_tensors: Tuple[torch.Tensor, torch.Tensor], dtype, device: str) -> None:
    x, y = input_tensors
    output = vsi(x.to(device=device, dtype=dtype), y.to(device=device, dtype=dtype))
    assert output.dtype == dtype


# ================== Test class: `VSILoss` =================
def test_vsi_loss(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    x, y = input_tensors
    x.requires_grad_()
    loss = VSILoss(data_range=1.)(x.to(device), y.to(device))
    loss.backward()
    assert torch.isfinite(x.grad).all(), \
        f'Expected finite gradient values after back propagation, got {x.grad}'


def test_vsi_loss_zero_for_equal_input(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    x, _ = input_tensors
    y = x.clone()
    x.requires_grad_()
    loss = VSILoss(data_range=1.)(x.to(device), y.to(device))
    assert torch.isclose(loss, torch.zeros_like(loss)), \
        f'Expected loss equals zero for identical inputs, got {loss}'
