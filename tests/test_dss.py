import torch
import pytest

from piq import dss, DSSLoss
from skimage.io import imread
from typing import Tuple


def _rgb2gray(x):
    return 0.2989 * x[:, :, 0] + 0.5870 * x[:, :, 1] + 0.1140 * x[:, :, 2]


# ================== Test function: `dss` ==================
def test_dss_to_be_one_for_identical_inputs(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction, _ = input_tensors
    index = dss(prediction.to(device), prediction.to(device), data_range=1., reduction='none')
    index_255 = dss(prediction.to(device) * 255, prediction.to(device) * 255, data_range=255, reduction='none')
    assert torch.allclose(index, torch.ones_like(index, device=device)), \
        f'Expected index to be equal 1, got {index}'
    assert torch.allclose(index_255, torch.ones_like(index_255, device=device)), \
        f'Expected index to be equal 1, got {index_255}'


def test_dss_symmetry(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction, target = input_tensors
    result = dss(prediction.to(device), target.to(device), data_range=1., reduction='none')
    result_sym = dss(target.to(device), prediction.to(device), data_range=1., reduction='none')
    assert torch.allclose(result_sym, result), f'Expected the same results, got {result} and {result_sym}'


def test_dss_zeros_ones_inputs(device: str) -> None:
    zeros = torch.zeros(1, 3, 128, 128, device=device)
    ones = torch.ones(1, 3, 128, 128, device=device)
    dss_ones = dss(ones, ones, data_range=1.)
    assert torch.isfinite(dss_ones).all(), f'Expected finite value for ones tensos, got {dss_ones}'
    dss_zeros_ones = dss(zeros, ones, data_range=1.)
    assert torch.isfinite(dss_zeros_ones).all(), \
        f'Expected finite value for zeros and ones tensos, got {dss_zeros_ones}'


def test_dss_raises_if_tensors_have_different_shapes(device: str) -> None:
    prediction = torch.zeros(1, 3, 64, 66, device=device)
    target = torch.ones(1, 3, 62, 66, device=device)
    with pytest.raises(AssertionError):
        dss(prediction, target)


def test_dss_raises_if_tensors_have_different_types(x: torch.Tensor) -> None:
    wrong_type_prediction = list(range(10))
    with pytest.raises(AssertionError):
        dss(wrong_type_prediction, x)


def test_dss_raises_if_incorrect_value(x: torch.Tensor, y: torch.Tensor, device: str) -> None:
    # DCT & kernel size
    size = max(x.size(-1), x.size(-2)) + 2
    with pytest.raises(ValueError):
        dss(x.to(device), y.to(device), data_range=1., dct_size=size)
    with pytest.raises(ValueError):
        dss(x.to(device), y.to(device), data_range=1., kernel_size=size)
    # Sigmas
    with pytest.raises(ValueError):
        dss(x.to(device), y.to(device), data_range=1., sigma_weight=0)
    with pytest.raises(ValueError):
        dss(x.to(device), y.to(device), data_range=1., sigma_similarity=0)
    # Percentile
    for percentile in [-0.5, 0, 0.01, 0.5, 1, 1.5]:
        if percentile <= 0 or percentile > 1:
            with pytest.raises(ValueError):
                dss(x.to(device), y.to(device), data_range=1., percentile=percentile)
        else:
            dss_result = dss(x.to(device), y.to(device), data_range=1., percentile=percentile,
                             reduction='none')
            assert torch.gt(dss_result, 0).all() and torch.le(dss_result, 1).all(), \
                'Out of bounds result'


def test_dss_supports_different_data_ranges(x: torch.Tensor, y: torch.Tensor, device: str) -> None:
    prediction_255 = (x * 255).type(torch.uint8)
    target_255 = (y * 255).type(torch.uint8)
    measure_255 = dss(prediction_255.to(device), target_255.to(device), data_range=255)
    measure = dss((prediction_255 / 255.).to(device), (target_255 / 255.).to(device))

    diff = torch.abs(measure_255 - measure)
    assert diff <= 1e-6, f'Result for same tensor with different data_range should be the same, got {diff}'


def test_dss_modes(x: torch.Tensor, y: torch.Tensor, device: str) -> None:
    for reduction in ['mean', 'sum']:
        measure = dss(x.to(device), y.to(device), reduction=reduction)
        assert measure.dim() == 0, f'DSS with `mean` reduction must return 1 number, got {len(measure)}'

    measure = dss(x.to(device), y.to(device), reduction='none')
    assert len(measure) == x.size(0), \
        f'DSS with `none` reduction must have length equal to number of images, got {len(measure)}'

    for reduction in ['DEADBEEF', 'random']:
        with pytest.raises(ValueError):
            dss(x.to(device), y.to(device), reduction=reduction)


def test_dss_compare_with_matlab(device: str) -> None:
    # greyscale image
    prediction = torch.tensor(imread('tests/assets/goldhill.gif')).unsqueeze(0).unsqueeze(0)
    target = torch.tensor(imread('tests/assets/goldhill_jpeg.gif')).unsqueeze(0).unsqueeze(0)
    predicted_score = dss(prediction.to(device), target.to(device), data_range=255, reduction='none')
    target_score = torch.tensor([0.45217181]).to(predicted_score)
    assert torch.isclose(predicted_score, target_score, atol=1e-3), \
        f'Expected result similar to MATLAB, got diff{predicted_score - target_score}'

    # color image : use same rgb2gray formula as matlab
    prediction = torch.tensor(_rgb2gray(imread('tests/assets/I01.BMP'))).to(torch.uint8).unsqueeze(0).unsqueeze(0)
    target = torch.tensor(_rgb2gray(imread('tests/assets/i01_01_5.bmp'))).to(torch.uint8).unsqueeze(0).unsqueeze(0)
    predicted_score = dss(prediction.to(device), target.to(device), data_range=255, reduction='none')
    target_score = torch.tensor([0.77177436]).to(predicted_score)
    assert torch.isclose(predicted_score, target_score, atol=1e-4), \
        f'Expected result similar to MATLAB, got diff{predicted_score - target_score}'


def test_dss_raises_if_input_is_not_4d(x: torch.Tensor, y: torch.Tensor) -> None:
    x_2d, y_2d = x[0, 0, ...], y[0, 0, ...]
    with pytest.raises(AssertionError):
        dss(x_2d, y_2d)

    x_3d, y_3d = x[0], y[0]
    with pytest.raises(AssertionError):
        dss(x_3d, y_3d)

    x_5d, y_5d = x.unsqueeze(0), y.unsqueeze(0)
    with pytest.raises(AssertionError):
        dss(x_5d, y_5d)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64],
)
def test_dss_preserves_dtype(input_tensors: torch.Tensor, dtype, device: str) -> None:
    x, y = input_tensors
    output = dss(x.to(device=device, dtype=dtype), y.to(device=device, dtype=dtype))
    assert output.dtype == dtype


# ================== Test class: `DSSLoss` =================
def test_dss_loss(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction, target = input_tensors
    prediction.requires_grad_()
    loss = DSSLoss(data_range=1.)(prediction.to(device), target.to(device))
    loss.backward()
    assert torch.isfinite(prediction.grad).all(), \
        f'Expected finite gradient values after back propagation, got {prediction.grad}'


def test_dss_loss_zero_for_equal_input(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction, _ = input_tensors
    target = prediction.clone()
    prediction.requires_grad_()
    loss = DSSLoss(data_range=1.)(prediction.to(device), target.to(device))
    assert torch.isclose(loss, torch.zeros_like(loss)), \
        f'Expected loss equals zero for identical inputs, got {loss}'
