from typing import Any, Tuple
import torch
import pytest
from skimage.io import imread
from contextlib import contextmanager

from piq import srsim, SRSIMLoss


# ================== Test function: `srsrim` ==================
def test_srsim_to_be_one_for_identical_inputs(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction, _ = input_tensors
    print(prediction.size())
    index = srsim(prediction.to(device), prediction.to(device), data_range=1., reduction='none')
    index_255 = srsim(prediction.to(device) * 255, prediction.to(device) * 255, data_range=255, reduction='none')
    assert torch.allclose(index, torch.ones_like(index, device=device)), \
        f'Expected index to be equal 1, got {index}'
    assert torch.allclose(index_255, torch.ones_like(index_255, device=device)), \
        f'Expected index to be equal 1, got {index_255}'


def test_srsim_symmetry(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction, target = input_tensors
    result = srsim(prediction.to(device), target.to(device), data_range=1., reduction='none')
    result_sym = srsim(target.to(device), prediction.to(device), data_range=1., reduction='none')
    assert torch.allclose(result_sym, result), f'Expected the same results, got {result} and {result_sym}'


def test_srsim_zeros_ones_inputs(device: str) -> None:
    zeros = torch.zeros(1, 3, 128, 128, device=device)
    ones = torch.ones(1, 3, 128, 128, device=device)
    srsim_zeros = srsim(zeros, zeros, data_range=1.)
    assert torch.isfinite(srsim_zeros).all(), f'Expected finite value for zeros tensors, got {srsim_zeros}'
    srsim_ones = srsim(ones, ones, data_range=1.)
    assert torch.isfinite(srsim_ones).all(), f'Expected finite value for ones tensos, got {srsim_ones}'
    srsim_zeros_ones = srsim(zeros, ones, data_range=1.)
    assert torch.isfinite(srsim_zeros_ones).all(), \
        f'Expected finite value for zeros and ones tensos, got {srsim_zeros_ones}'


def test_srsim_compare_with_matlab(device: str) -> None:
    # Greyscale image
    prediction = torch.tensor(imread('tests/assets/goldhill.gif'))
    target = torch.tensor(imread('tests/assets/goldhill_jpeg.gif'))
    predicted_score = srsim(prediction.to(device), target.to(device), data_range=255, reduction='none')
    target_score = torch.tensor([0.94623509]).to(predicted_score) # from matlab code
    assert torch.allclose(predicted_score, target_score), f'Expected result similar to MATLAB,' \
                                                          f'got diff{predicted_score - target_score}'
    # RBG image
    prediction = torch.tensor(imread('tests/assets/I01.BMP')).permute(2, 0, 1)
    target = torch.tensor(imread('tests/assets/i01_01_5.bmp')).permute(2, 0, 1)
    predicted_score = srsim(prediction.to(device), target.to(device), data_range=255, reduction='none')
    target_score = torch.tensor([0.96667468]).to(predicted_score) # from matlab code
    assert torch.allclose(predicted_score, target_score), f'Expected result similar to MATLAB,' \
                                                          f'got diff{predicted_score - target_score}'

def test_srsim_chromatic(device: str) -> None:
    # RBG image
    prediction = torch.tensor(imread('tests/assets/I01.BMP')).permute(2, 0, 1)
    target = torch.tensor(imread('tests/assets/i01_01_5.bmp')).permute(2, 0, 1)
    predicted_score = srsim(prediction.to(device), target.to(device), data_range=255, chromatic=True, reduction='none')
    target_score = torch.tensor([0.9553047]).to(predicted_score)
    assert torch.allclose(predicted_score, target_score), f'Expected result for chromatic version,' \
                                                          f'got diff{predicted_score - target_score}'

# ================== Test class: `srsimLoss` =================
# def test_fsim_loss_reduction(prediction: torch.Tensor, target: torch.Tensor) -> None:
#     loss = SRSIMLoss(reduction='mean')
#     measure = loss(prediction, target)
#     assert measure.dim() == 0, f'FSIM with `mean` reduction must return 1 number, got {len(measure)}'

#     loss = SRSIMLoss(reduction='sum')
#     measure = loss(prediction, target)
#     assert measure.dim() == 0, f'FSIM with `mean` reduction must return 1 number, got {len(measure)}'

#     loss = SRSIMLoss(reduction='none')
#     measure = loss(prediction, target)
#     assert len(measure) == prediction.size(0), \
#         f'FSIM with `none` reduction must have length equal to number of images, got {len(measure)}'

#     loss = SRSIMLoss(reduction='random string')
#     with pytest.raises(KeyError):
#         loss(prediction, target)

def test_srsim_loss(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction, target = input_tensors
    prediction.requires_grad_()
    loss = SRSIMLoss(data_range=1.)(prediction.to(device), target.to(device))
    loss.backward()
    assert torch.isfinite(prediction.grad).all(), \
        f'Expected finite gradient values after back propagation, got {prediction.grad}'


# def test_srsim_loss_zero_for_equal_input(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
#     prediction, _ = input_tensors
#     target = prediction.clone()
#     prediction.requires_grad_()
#     loss = SRSIMLoss(data_range=1.)(prediction.to(device), target.to(device))
#     assert torch.isclose(loss, torch.zeros_like(loss)), \
#         f'Expected loss equals zero for identical inputs, got {loss}'
