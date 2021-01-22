import torch
import itertools
import pytest
import tensorflow as tf
from piq import SSIMLoss, ssim
from typing import Tuple, List, Any
from skimage.io import imread
from contextlib import contextmanager


@contextmanager
def raise_nothing(enter_result=None):
    yield enter_result


@pytest.fixture(scope='module')
def prediction() -> torch.Tensor:
    return torch.rand(3, 3, 161, 161)


@pytest.fixture(scope='module')
def target() -> torch.Tensor:
    return torch.rand(3, 3, 161, 161)


@pytest.fixture(params=[(3, 3, 161, 161), (3, 3, 161, 161, 2)], scope='module')
def prediction_target_4d_5d(request: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.rand(request.param), torch.rand(request.param)


@pytest.fixture(params=[(3, 3, 161, 161), (3, 3, 161, 161, 2)], scope='module')
def ones_zeros_4d_5d(request: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ones(request.param), torch.zeros(request.param)


@pytest.fixture(scope='module')
def test_images() -> List[Tuple[torch.Tensor, torch.Tensor]]:
    prediction_grey = torch.tensor(imread('tests/assets/goldhill_jpeg.gif')).unsqueeze(0).unsqueeze(0)
    target_grey = torch.tensor(imread('tests/assets/goldhill.gif')).unsqueeze(0).unsqueeze(0)
    prediction_rgb = torch.tensor(imread('tests/assets/I01.BMP')).permute(2, 0, 1).unsqueeze(0)
    target_rgb = torch.tensor(imread('tests/assets/i01_01_5.bmp')).permute(2, 0, 1).unsqueeze(0)
    return [(prediction_grey, target_grey), (prediction_rgb, target_rgb)]


@pytest.fixture(params=[[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], [0.0448, 0.2856, 0.3001]], scope='module')
def scale_weights(request: Any) -> List:
    return request.param


# ================== Test function: `ssim` ==================
def test_ssim_symmetry(prediction_target_4d_5d: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction = prediction_target_4d_5d[0].to(device)
    target = prediction_target_4d_5d[1].to(device)
    measure = ssim(prediction, target, data_range=1., reduction='none')
    reverse_measure = ssim(target, prediction, data_range=1., reduction='none')
    assert torch.allclose(measure, reverse_measure), f'Expect: SSIM(a, b) == SSIM(b, a), ' \
                                                     f'got {measure} != {reverse_measure}'


def test_ssim_measure_is_one_for_equal_tensors(target: torch.Tensor, device: str) -> None:
    target = target.to(device)
    prediction = target.clone()
    measure = ssim(prediction, target, data_range=1., reduction='none')
    assert torch.allclose(measure, torch.ones_like(measure)), f'If equal tensors are passed SSIM must be equal to 1 ' \
                                                              f'(considering floating point error up to 1 * 10^-6), '\
                                                              f'got {measure}'


def test_ssim_reduction(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    for mode in ['mean', 'sum', 'none']:
        ssim(prediction.to(device), target.to(device), reduction=mode)

    for mode in [None, 'n', 2]:
        with pytest.raises(KeyError):
            ssim(prediction.to(device), target.to(device), reduction=mode)
            

def test_ssim_returns_full(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    prediction = prediction.to(device)
    target = target.to(device)
    assert len(ssim(prediction, target, full=True)) == 2, "Expected 2 output values, got 1"
        

def test_ssim_measure_is_less_or_equal_to_one(ones_zeros_4d_5d: Tuple[torch.Tensor, torch.Tensor],
                                              device: str) -> None:
    # Create two maximally different tensors.
    ones = ones_zeros_4d_5d[0].to(device)
    zeros = ones_zeros_4d_5d[1].to(device)
    measure = ssim(ones, zeros, data_range=1., reduction='none')
    assert torch.le(measure, 1).all(), f'SSIM must be <= 1, got {measure}'


def test_ssim_raises_if_tensors_have_different_shapes(prediction_target_4d_5d: Tuple[torch.Tensor,
                                                                                     torch.Tensor], device) -> None:
    target = prediction_target_4d_5d[1].to(device)
    dims = [[3], [2, 3], [161, 162], [161, 162]]
    if target.dim() == 5:
        dims += [[2, 3]]
    for size in list(itertools.product(*dims)):
        wrong_shape_prediction = torch.rand(size).to(target)
        if wrong_shape_prediction.size() == target.size():
            try:
                ssim(wrong_shape_prediction, target)
            except Exception as e:
                pytest.fail(f"Unexpected error occurred: {e}")
        else:
            with pytest.raises(AssertionError):
                ssim(wrong_shape_prediction, target)


def test_ssim_raises_if_tensors_have_different_types(target: torch.Tensor) -> None:
    wrong_type_prediction = list(range(10))
    with pytest.raises(AssertionError):
        ssim(wrong_type_prediction, target)


def test_ssim_check_available_dimensions() -> None:
    custom_prediction = torch.rand(256, 256)
    custom_target = torch.rand(256, 256)
    for _ in range(10):
        if custom_prediction.dim() < 5:
            try:
                ssim(custom_prediction, custom_target)
            except Exception as e:
                pytest.fail(f"Unexpected error occurred: {e}")
        else:
            with pytest.raises(AssertionError):
                ssim(custom_prediction, custom_target)
        custom_prediction.unsqueeze_(0)
        custom_target.unsqueeze_(0)


def test_ssim_check_kernel_size_is_passed(prediction_target_4d_5d: Tuple[torch.Tensor, torch.Tensor],
                                          device: str) -> None:
    prediction = prediction_target_4d_5d[0].to(device)
    target = prediction_target_4d_5d[1].to(device)
    kernel_sizes = list(range(0, 50))
    for kernel_size in kernel_sizes:
        if kernel_size % 2:
            ssim(prediction, target, kernel_size=kernel_size)
        else:
            with pytest.raises(AssertionError):
                ssim(prediction, target, kernel_size=kernel_size)


def test_ssim_raises_if_kernel_size_greater_than_image(prediction_target_4d_5d: Tuple[torch.Tensor, torch.Tensor],
                                                       device: str) -> None:
    prediction = prediction_target_4d_5d[0].to(device)
    target = prediction_target_4d_5d[1].to(device)
    kernel_size = 11
    wrong_size_prediction = prediction[:, :, :kernel_size - 1, :kernel_size - 1]
    wrong_size_target = target[:, :, :kernel_size - 1, :kernel_size - 1]
    with pytest.raises(ValueError):
        ssim(wrong_size_prediction, wrong_size_target, kernel_size=kernel_size)


def test_ssim_raise_if_wrong_value_is_estimated(test_images: Tuple[torch.Tensor, torch.Tensor],
                                                device: str) -> None:
    for prediction, target in test_images:
        piq_ssim = ssim(prediction.to(device), target.to(device), kernel_size=11, kernel_sigma=1.5, data_range=255,
                        reduction='none', downsample=False)
        tf_prediction = tf.convert_to_tensor(prediction.permute(0, 2, 3, 1).numpy())
        tf_target = tf.convert_to_tensor(target.permute(0, 2, 3, 1).numpy())
        with tf.device('/CPU'):
            tf_ssim = torch.tensor(tf.image.ssim(tf_prediction, tf_target, max_val=255).numpy()).to(piq_ssim)
        match_accuracy = 2e-4 + 1e-8
        assert torch.allclose(piq_ssim, tf_ssim, rtol=0, atol=match_accuracy), \
            f'The estimated value must be equal to tensorflow provided one' \
            f'(considering floating point operation error up to {match_accuracy}), ' \
            f'got difference {(piq_ssim - tf_ssim).abs()}'


@pytest.mark.parametrize(
    "data_range", [128, 255],
)
def test_ssim_supports_different_data_ranges(
        input_tensors: Tuple[torch.Tensor, torch.Tensor], data_range, device: str) -> None:
    prediction, target = input_tensors
    prediction_scaled = (prediction * data_range).type(torch.uint8)
    target_scaled = (target * data_range).type(torch.uint8)

    measure_scaled = ssim(prediction_scaled.to(device), target_scaled.to(device), data_range=data_range)
    measure = ssim(
        prediction_scaled.to(device) / float(data_range),
        target_scaled.to(device) / float(data_range),
        data_range=1.0
    )
    diff = torch.abs(measure_scaled - measure)
    assert diff <= 1e-6, f'Result for same tensor with different data_range should be the same, got {diff}'


def test_ssim_fails_for_incorrect_data_range(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    # Scale to [0, 255]
    prediction_scaled = (prediction * 255).type(torch.uint8)
    target_scaled = (target * 255).type(torch.uint8)
    with pytest.raises(AssertionError):
        ssim(prediction_scaled.to(device), target_scaled.to(device), data_range=1.0)


# ================== Test class: `SSIMLoss` ==================
def test_ssim_loss_grad(prediction_target_4d_5d: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction = prediction_target_4d_5d[0].to(device)
    target = prediction_target_4d_5d[1].to(device)
    prediction.requires_grad_(True)
    loss = SSIMLoss(data_range=1.)(prediction, target).mean()
    loss.backward()
    assert torch.isfinite(prediction.grad).all(), f'Expected finite gradient values, got {prediction.grad}'


def test_ssim_loss_symmetry(prediction_target_4d_5d: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction = prediction_target_4d_5d[0].to(device)
    target = prediction_target_4d_5d[1].to(device)
    loss = SSIMLoss()
    loss_value = loss(prediction, target)
    reverse_loss_value = loss(target, prediction)
    assert torch.allclose(loss_value, reverse_loss_value), \
        f'Expect: SSIMLoss(a, b) == SSIMLoss(b, a), got {loss_value} != {reverse_loss_value}'


def test_ssim_loss_equality(target: torch.Tensor, device: str) -> None:
    target = target.to(device)
    prediction = target.clone()
    loss = SSIMLoss()(prediction, target)
    assert torch.allclose(loss, torch.zeros_like(loss)), \
        f'If equal tensors are passed SSIM loss must be equal to 0 '\
        f'(considering floating point operation error up to 1 * 10^-6), got {loss}'


def test_ssim_loss_is_less_or_equal_to_one(ones_zeros_4d_5d: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    # Create two maximally different tensors.
    ones = ones_zeros_4d_5d[0].to(device)
    zeros = ones_zeros_4d_5d[1].to(device)
    loss = SSIMLoss()(ones, zeros)
    assert (loss <= 1).all(), f'SSIM loss must be <= 1, got {loss}'


def test_ssim_loss_raises_if_tensors_have_different_shapes(prediction_target_4d_5d: Tuple[torch.Tensor,
                                                                                          torch.Tensor],
                                                           device) -> None:
    target = prediction_target_4d_5d[1].to(device)
    dims = [[3], [2, 3], [161, 162], [161, 162]]
    if target.dim() == 5:
        dims += [[2, 3]]
    for size in list(itertools.product(*dims)):
        wrong_shape_prediction = torch.rand(size).to(target)
        if wrong_shape_prediction.size() == target.size():
            SSIMLoss()(wrong_shape_prediction, target)
        else:
            with pytest.raises(AssertionError):
                SSIMLoss()(wrong_shape_prediction, target)


def test_ssim_loss_check_available_dimensions() -> None:
    custom_prediction = torch.rand(256, 256)
    custom_target = torch.rand(256, 256)
    for _ in range(10):
        if custom_prediction.dim() < 5:
            try:
                SSIMLoss()(custom_prediction, custom_target)
            except Exception as e:
                pytest.fail(f"Unexpected error occurred: {e}")
        else:
            with pytest.raises(AssertionError):
                SSIMLoss()(custom_prediction, custom_target)
        custom_prediction.unsqueeze_(0)
        custom_target.unsqueeze_(0)


def test_ssim_loss_raises_if_tensors_have_different_types(target: torch.Tensor) -> None:
    wrong_type_prediction = list(range(10))
    with pytest.raises(AssertionError):
        SSIMLoss()(wrong_type_prediction, target)


def test_ssim_loss_check_kernel_size_is_passed(prediction: torch.Tensor, target: torch.Tensor) -> None:
    kernel_sizes = list(range(0, 50))
    for kernel_size in kernel_sizes:
        if kernel_size % 2:
            SSIMLoss(kernel_size=kernel_size)(prediction, target)
        else:
            with pytest.raises(AssertionError):
                SSIMLoss(kernel_size=kernel_size)(prediction, target)


def test_ssim_loss_raises_if_kernel_size_greater_than_image(prediction_target_4d_5d: Tuple[torch.Tensor, torch.Tensor],
                                                            device: str) -> None:
    prediction = prediction_target_4d_5d[0].to(device)
    target = prediction_target_4d_5d[1].to(device)
    kernel_size = 11
    wrong_size_prediction = prediction[:, :, :kernel_size - 1, :kernel_size - 1]
    wrong_size_target = target[:, :, :kernel_size - 1, :kernel_size - 1]
    with pytest.raises(ValueError):
        SSIMLoss(kernel_size=kernel_size)(wrong_size_prediction, wrong_size_target)


def test_ssim_loss_raise_if_wrong_value_is_estimated(test_images: Tuple[torch.Tensor, torch.Tensor],
                                                     device: str) -> None:
    for x, y in test_images:
        ssim_loss = SSIMLoss(
            kernel_size=11, kernel_sigma=1.5, data_range=255, downsample=False)(x.to(device), y.to(device))
        tf_x = tf.convert_to_tensor(x.permute(0, 2, 3, 1).numpy())
        tf_y = tf.convert_to_tensor(y.permute(0, 2, 3, 1).numpy())
        with tf.device('/CPU'):
            tf_ssim = torch.tensor(tf.image.ssim(tf_x, tf_y, max_val=255).numpy()).mean().to(device)
        match_accuracy = 2e-4 + 1e-8
        assert torch.isclose(ssim_loss, 1. - tf_ssim, rtol=0, atol=match_accuracy), \
            f'The estimated value must be equal to tensorflow provided one' \
            f'(considering floating point operation error up to {match_accuracy}), ' \
            f'got difference {(ssim_loss - 1. + tf_ssim).abs()}'


def test_ssim_simmular_to_matlab_implementation():
    # Greyscale images
    goldhill = torch.tensor(imread('tests/assets/goldhill.gif'))
    goldhill_jpeg = torch.tensor(imread('tests/assets/goldhill_jpeg.gif'))

    score = ssim(goldhill_jpeg, goldhill, data_range=255, reduction='none')
    # Output of http://www.cns.nyu.edu/~lcv/ssim/ssim.m
    score_baseline = torch.tensor(0.8202)

    assert torch.isclose(score, score_baseline, atol=1e-4), \
        f'Expected PyTorch score to be equal to MATLAB prediction. Got {score} and {score_baseline}'

    # RGB images
    I01 = torch.tensor(imread('tests/assets/I01.BMP')).permute(2, 0, 1)
    i1_01_5 = torch.tensor(imread('tests/assets/i01_01_5.bmp')).permute(2, 0, 1)

    score = ssim(i1_01_5, I01, data_range=255, reduction='none')
    # Output of http://www.cns.nyu.edu/~lcv/ssim/ssim.m
    # score_baseline = torch.tensor(0.7820)
    score_baseline = torch.tensor(0.7842)

    assert torch.isclose(score, score_baseline, atol=1e-2), \
        f'Expected PyTorch score to be equal to MATLAB prediction. Got {score} and {score_baseline}'
