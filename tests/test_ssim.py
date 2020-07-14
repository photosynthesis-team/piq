import torch
import itertools
import pytest
import tensorflow as tf
from piq import SSIMLoss, MultiScaleSSIMLoss, ssim, multi_scale_ssim
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
                                                              f'got {measure + 1}'


@pytest.mark.parametrize(
    "reduction,full,expectation",
    [('mean', False, raise_nothing()),
     ('sum', False, raise_nothing()),
     ('none', False, raise_nothing()),
     ('none', True, raise_nothing()),
     ('reduction', False, pytest.raises(KeyError))]
)
def test_ssim_reduction_and_full(reduction: str, full: bool, expectation: Any,
                                 prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    prediction = prediction.to(device)
    target = target.to(device)
    with expectation:
        ssim(prediction, target, data_range=1., reduction=reduction, full=full)


def test_ssim_measure_is_less_or_equal_to_one(ones_zeros_4d_5d: Tuple[torch.Tensor, torch.Tensor],
                                              device: str) -> None:
    # Create two maximally different tensors.
    ones = ones_zeros_4d_5d[0].to(device)
    zeros = ones_zeros_4d_5d[1].to(device)
    measure = ssim(ones, zeros, data_range=1., reduction='none')
    assert (measure <= 1).all(), f'SSIM must be <= 1, got {measure}'


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
                        reduction='none')
        tf_prediction = tf.convert_to_tensor(prediction.permute(0, 2, 3, 1).numpy())
        tf_target = tf.convert_to_tensor(target.permute(0, 2, 3, 1).numpy())
        with tf.device('/CPU'):
            tf_ssim = torch.tensor(tf.image.ssim(tf_prediction, tf_target, max_val=255).numpy()).to(piq_ssim)
        match_accuracy = 2e-5 + 1e-8
        assert torch.allclose(piq_ssim, tf_ssim, rtol=0, atol=match_accuracy), \
            f'The estimated value must be equal to tensorflow provided one' \
            f'(considering floating point operation error up to {match_accuracy}), ' \
            f'got difference {(piq_ssim - tf_ssim).abs()}'


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
    for prediction, target in test_images:
        ssim_loss = SSIMLoss(kernel_size=11, kernel_sigma=1.5, data_range=255, reduction='mean')(prediction.to(device),
                                                                                                 target.to(device))
        tf_prediction = tf.convert_to_tensor(prediction.permute(0, 2, 3, 1).numpy())
        tf_target = tf.convert_to_tensor(target.permute(0, 2, 3, 1).numpy())
        with tf.device('/CPU'):
            tf_ssim = torch.tensor(tf.image.ssim(tf_prediction, tf_target, max_val=255).numpy()).mean().to(device)
        match_accuracy = 2e-5 + 1e-8
        assert torch.isclose(ssim_loss, 1. - tf_ssim, rtol=0, atol=match_accuracy), \
            f'The estimated value must be equal to tensorflow provided one' \
            f'(considering floating point operation error up to {match_accuracy}), ' \
            f'got difference {(ssim_loss - 1. + tf_ssim).abs()}'


# ================== Test function: `multi_scale_ssim` ==================
def test_multi_scale_ssim_symmetry(prediction_target_4d_5d: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction = prediction_target_4d_5d[0].to(device)
    target = prediction_target_4d_5d[1].to(device)
    measure = multi_scale_ssim(prediction, target, data_range=1., reduction='none')
    reverse_measure = multi_scale_ssim(target, prediction, data_range=1., reduction='none')
    assert torch.allclose(measure, reverse_measure), f'Expect: MS-SSIM(a, b) == MSSSIM(b, a), '\
                                                     f'got {measure} != {reverse_measure}'


def test_multi_scale_ssim_measure_is_one_for_equal_tensors(target: torch.Tensor, device: str) -> None:
    target = target.to(device)
    prediction = target.clone()
    measure = multi_scale_ssim(prediction, target, data_range=1.)
    assert torch.allclose(measure, torch.ones_like(measure)), \
        f'If equal tensors are passed MS-SSIM must be equal to 1 ' \
        f'(considering floating point operation error up to 1 * 10^-6), got {measure + 1}'


def test_multi_scale_ssim_measure_is_less_or_equal_to_one(ones_zeros_4d_5d: Tuple[torch.Tensor, torch.Tensor],
                                                          device: str) -> None:
    # Create two maximally different tensors.
    ones = ones_zeros_4d_5d[0].to(device)
    zeros = ones_zeros_4d_5d[1].to(device)
    measure = multi_scale_ssim(ones, zeros, data_range=1.)
    assert (measure <= 1).all(), f'MS-SSIM must be <= 1, got {measure}'


def test_multi_scale_ssim_raises_if_tensors_have_different_shapes(prediction_target_4d_5d: Tuple[torch.Tensor,
                                                                                                 torch.Tensor],
                                                                  device: str) -> None:
    target = prediction_target_4d_5d[1].to(device)
    dims = [[3], [2, 3], [161, 162], [161, 162]]
    if target.dim() == 5:
        dims += [[2, 3]]
    for size in list(itertools.product(*dims)):
        wrong_shape_prediction = torch.rand(size).to(target)
        if wrong_shape_prediction.size() == target.size():
            multi_scale_ssim(wrong_shape_prediction, target)
        else:
            with pytest.raises(AssertionError):
                multi_scale_ssim(wrong_shape_prediction, target)
    scale_weights = torch.rand(2, 2)
    with pytest.raises(AssertionError):
        multi_scale_ssim(prediction, target, scale_weights=scale_weights)


def test_multi_scale_ssim_check_available_dimensions() -> None:
    custom_prediction = torch.rand(256, 256)
    custom_target = torch.rand(256, 256)
    for _ in range(10):
        if custom_prediction.dim() < 5:
            try:
                multi_scale_ssim(custom_prediction, custom_target)
            except Exception as e:
                pytest.fail(f"Unexpected error occurred: {e}")
        else:
            with pytest.raises(AssertionError):
                multi_scale_ssim(custom_prediction, custom_target)
        custom_prediction.unsqueeze_(0)
        custom_target.unsqueeze_(0)


def test_multi_scale_ssim_raises_if_tensors_have_different_types(prediction: torch.Tensor,
                                                                 target: torch.Tensor) -> None:
    wrong_type_prediction = list(range(10))
    with pytest.raises(AssertionError):
        multi_scale_ssim(wrong_type_prediction, target)
    wrong_type_scale_weights = True
    with pytest.raises(AssertionError):
        multi_scale_ssim(prediction, target, scale_weights=wrong_type_scale_weights)


def test_multi_scale_ssim_check_kernel_size_is_passed(prediction: torch.Tensor, target: torch.Tensor) -> None:
    kernel_sizes = list(range(0, 13))
    for kernel_size in kernel_sizes:
        if kernel_size % 2:
            multi_scale_ssim(prediction, target, kernel_size=kernel_size)
        else:
            with pytest.raises(AssertionError):
                multi_scale_ssim(prediction, target, kernel_size=kernel_size)


def test_ms_ssim_raises_if_kernel_size_greater_than_image(prediction_target_4d_5d: Tuple[torch.Tensor, torch.Tensor],
                                                          device: str) -> None:
    prediction = prediction_target_4d_5d[0].to(device)
    target = prediction_target_4d_5d[1].to(device)
    kernel_size = 11
    levels = 5
    min_size = (kernel_size - 1) * 2 ** (levels - 1) + 1
    wrong_size_prediction = prediction[:, :, :min_size - 1, :min_size - 1]
    wrong_size_target = target[:, :, :min_size - 1, :min_size - 1]
    with pytest.raises(ValueError):
        multi_scale_ssim(wrong_size_prediction, wrong_size_target, kernel_size=kernel_size)


def test_multi_scale_ssim_raise_if_wrong_value_is_estimated(test_images: Tuple[torch.Tensor, torch.Tensor],
                                                            scale_weights: List, device: str) -> None:
    for prediction, target in test_images:
        piq_ms_ssim = multi_scale_ssim(prediction.to(device), target.to(device), kernel_size=11, kernel_sigma=1.5,
                                       data_range=255, reduction='none', scale_weights=scale_weights)
        tf_prediction = tf.convert_to_tensor(prediction.permute(0, 2, 3, 1).numpy())
        tf_target = tf.convert_to_tensor(target.permute(0, 2, 3, 1).numpy())
        with tf.device('/CPU'):
            tf_ms_ssim = torch.tensor(tf.image.ssim_multiscale(tf_prediction, tf_target, max_val=255,
                                                               power_factors=scale_weights).numpy()).to(device)
        number_of_weights = 5.
        match_accuracy = number_of_weights * 1e-5 + 1e-8
        assert torch.allclose(piq_ms_ssim, tf_ms_ssim, rtol=0, atol=match_accuracy), \
            f'The estimated value must be equal to tensorflow provided one' \
            f'(considering floating point operation error up to {match_accuracy}), ' \
            f'got difference {(piq_ms_ssim - tf_ms_ssim).abs()}'


# ================== Test class: `MultiScaleSSIMLoss` ==================
def test_multi_scale_ssim_loss_grad(prediction_target_4d_5d: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction = prediction_target_4d_5d[0].to(device)
    target = prediction_target_4d_5d[1].to(device)
    prediction.requires_grad_()
    loss = MultiScaleSSIMLoss(data_range=1.)(prediction, target).mean()
    loss.backward()
    assert torch.isfinite(prediction.grad).all(), f'Expected finite gradient values, got {prediction.grad}'


def test_multi_scale_ssim_loss_symmetry(prediction_target_4d_5d: Tuple[torch.Tensor, torch.Tensor],
                                        device: str) -> None:
    prediction = prediction_target_4d_5d[0].to(device)
    target = prediction_target_4d_5d[1].to(device)
    loss = MultiScaleSSIMLoss()
    loss_value = loss(prediction, target)
    reverse_loss_value = loss(target, prediction)
    assert (loss_value == reverse_loss_value).all(), \
        f'Expect: MS-SSIM(a, b) == MS-SSIM(b, a), got {loss_value} != {reverse_loss_value}'


def test_multi_scale_ssim_loss_equality(target: torch.Tensor, device: str) -> None:
    target = target.to(device)
    prediction = target.clone()
    loss = MultiScaleSSIMLoss()(prediction, target)
    assert (loss.abs() <= 1e-6).all(), f'If equal tensors are passed SSIM loss must be equal to 0 ' \
                                       f'(considering floating point operation error up to 1 * 10^-6), got {loss}'


def test_multi_scale_ssim_loss_is_less_or_equal_to_one(ones_zeros_4d_5d: Tuple[torch.Tensor, torch.Tensor],
                                                       device: str) -> None:
    # Create two maximally different tensors.
    ones = ones_zeros_4d_5d[0].to(device)
    zeros = ones_zeros_4d_5d[1].to(device)
    loss = MultiScaleSSIMLoss()(ones, zeros)
    assert (loss <= 1).all(), f'MS-SSIM loss must be <= 1, got {loss}'


def test_multi_scale_ssim_loss_raises_if_tensors_have_different_shapes(prediction_target_4d_5d: Tuple[torch.Tensor,
                                                                                                      torch.Tensor],
                                                                       device: str) -> None:
    target = prediction_target_4d_5d[1].to(device)
    dims = [[3], [2, 3], [161, 162], [161, 162]]
    if target.dim() == 5:
        dims += [[2, 3]]
    for size in list(itertools.product(*dims)):
        wrong_shape_prediction = torch.rand(size).to(target)
        if wrong_shape_prediction.size() == target.size():
            MultiScaleSSIMLoss()(wrong_shape_prediction, target)
        else:
            with pytest.raises(AssertionError):
                MultiScaleSSIMLoss()(wrong_shape_prediction, target)
    scale_weights = torch.rand(2, 2)
    with pytest.raises(AssertionError):
        MultiScaleSSIMLoss(scale_weights=scale_weights)(prediction, target)


def test_multi_scale_ssim_loss_check_available_dimensions() -> None:
    custom_prediction = torch.rand(256, 256)
    custom_target = torch.rand(256, 256)
    for _ in range(10):
        if custom_prediction.dim() < 5:
            try:
                MultiScaleSSIMLoss()(custom_prediction, custom_target)
            except Exception as e:
                pytest.fail(f"Unexpected error occurred: {e}")
        else:
            with pytest.raises(AssertionError):
                MultiScaleSSIMLoss()(custom_prediction, custom_target)
        custom_prediction.unsqueeze_(0)
        custom_target.unsqueeze_(0)


def test_multi_scale_ssim_loss_raises_if_tensors_have_different_types(prediction: torch.Tensor,
                                                                      target: torch.Tensor) -> None:
    wrong_type_prediction = list(range(10))
    with pytest.raises(AssertionError):
        MultiScaleSSIMLoss()(wrong_type_prediction, target)
    wrong_type_scale_weights = True
    with pytest.raises(AssertionError):
        MultiScaleSSIMLoss(scale_weights=wrong_type_scale_weights)(prediction, target)


def test_multi_scale_ssim_loss_raises_if_wrong_kernel_size_is_passed(prediction: torch.Tensor,
                                                                     target: torch.Tensor) -> None:
    kernel_sizes = list(range(0, 13))
    for kernel_size in kernel_sizes:
        if kernel_size % 2:
            MultiScaleSSIMLoss(kernel_size=kernel_size)(prediction, target)
        else:
            with pytest.raises(AssertionError):
                MultiScaleSSIMLoss(kernel_size=kernel_size)(prediction, target)


def test_ms_ssim_loss_raises_if_kernel_size_greater_than_image(prediction_target_4d_5d: Tuple[torch.Tensor,
                                                                                              torch.Tensor],
                                                               device: str) -> None:
    prediction = prediction_target_4d_5d[0].to(device)
    target = prediction_target_4d_5d[1].to(device)
    kernel_size = 11
    levels = 5
    min_size = (kernel_size - 1) * 2 ** (levels - 1) + 1
    wrong_size_prediction = prediction[:, :, :min_size - 1, :min_size - 1]
    wrong_size_target = target[:, :, :min_size - 1, :min_size - 1]
    with pytest.raises(ValueError):
        MultiScaleSSIMLoss(kernel_size=kernel_size)(wrong_size_prediction, wrong_size_target)


def test_multi_scale_ssim_loss_raise_if_wrong_value_is_estimated(test_images: List, scale_weights: List,
                                                                 device: str) -> None:
    for prediction, target in test_images:
        piq_loss = MultiScaleSSIMLoss(kernel_size=11, kernel_sigma=1.5, data_range=255, scale_weights=scale_weights)
        piq_ms_ssim_loss = piq_loss(prediction.to(device), target.to(device))
        tf_prediction = tf.convert_to_tensor(prediction.permute(0, 2, 3, 1).numpy())
        tf_target = tf.convert_to_tensor(target.permute(0, 2, 3, 1).numpy())
        with tf.device('/CPU'):
            tf_ms_ssim = torch.tensor(tf.image.ssim_multiscale(tf_prediction, tf_target,
                                                               power_factors=scale_weights,
                                                               max_val=255).numpy()).mean().to(device)
        number_of_weights = len(piq_loss.scale_weights)
        match_accuracy = number_of_weights * 1e-5 + 1e-8
        assert torch.isclose(piq_ms_ssim_loss, 1. - tf_ms_ssim, rtol=0, atol=match_accuracy), \
            f'The estimated value must be equal to tensorflow provided one' \
            f'(considering floating point operation error up to {match_accuracy}), ' \
            f'got difference {(piq_ms_ssim_loss - 1. + tf_ms_ssim).abs()}'
