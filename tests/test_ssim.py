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
def x() -> torch.Tensor:
    return torch.rand(3, 3, 161, 161)


@pytest.fixture(scope='module')
def y() -> torch.Tensor:
    return torch.rand(3, 3, 161, 161)


@pytest.fixture(params=[(3, 3, 161, 161), (3, 3, 161, 161, 2)], scope='module')
def x_y_4d_5d(request: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.rand(request.param), torch.rand(request.param)


@pytest.fixture(params=[(3, 3, 161, 161), (3, 3, 161, 161, 2)], scope='module')
def ones_zeros_4d_5d(request: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ones(request.param), torch.zeros(request.param)


@pytest.fixture(scope='module')
def test_images() -> List[Tuple[torch.Tensor, torch.Tensor]]:
    x_grey = torch.tensor(imread('tests/assets/goldhill_jpeg.gif')).unsqueeze(0).unsqueeze(0)
    y_grey = torch.tensor(imread('tests/assets/goldhill.gif')).unsqueeze(0).unsqueeze(0)
    x_rgb = torch.tensor(imread('tests/assets/I01.BMP')).permute(2, 0, 1).unsqueeze(0)
    y_rgb = torch.tensor(imread('tests/assets/i01_01_5.bmp')).permute(2, 0, 1).unsqueeze(0)
    return [(x_grey, y_grey), (x_rgb, y_rgb)]


@pytest.fixture(params=[[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], [0.0448, 0.2856, 0.3001]], scope='module')
def scale_weights(request: Any) -> List:
    return request.param


# ================== Test function: `ssim` ==================
def test_ssim_symmetry(x_y_4d_5d, device: str) -> None:
    x = x_y_4d_5d[0].to(device)
    y = x_y_4d_5d[1].to(device)
    measure = ssim(x, y, data_range=1., reduction='none')
    reverse_measure = ssim(y, x, data_range=1., reduction='none')
    assert torch.allclose(measure, reverse_measure), f'Expect: SSIM(a, b) == SSIM(b, a), ' \
                                                     f'got {measure} != {reverse_measure}'


def test_ssim_measure_is_one_for_equal_tensors(y: torch.Tensor, device: str) -> None:
    y = y.to(device)
    x = y.clone()
    measure = ssim(x, y, data_range=1., reduction='none')
    assert torch.allclose(measure, torch.ones_like(measure)), f'If equal tensors are passed SSIM must be equal to 1 ' \
                                                              f'(considering floating point error up to 1 * 10^-6), '\
                                                              f'got {measure}'


def test_ssim_reduction(x: torch.Tensor, y: torch.Tensor, device: str) -> None:
    for mode in ['mean', 'sum', 'none']:
        ssim(x.to(device), y.to(device), reduction=mode)

    for mode in [None, 'n', 2]:
        with pytest.raises(ValueError):
            ssim(x.to(device), y.to(device), reduction=mode)
            

def test_ssim_returns_full(x: torch.Tensor, y: torch.Tensor, device: str) -> None:
    x = x.to(device)
    y = y.to(device)
    assert len(ssim(x, y, full=True)) == 2, "Expected 2 output values, got 1"
        

def test_ssim_measure_is_less_or_equal_to_one(ones_zeros_4d_5d: Tuple[torch.Tensor, torch.Tensor],
                                              device: str) -> None:
    # Create two maximally different tensors.
    ones = ones_zeros_4d_5d[0].to(device)
    zeros = ones_zeros_4d_5d[1].to(device)
    measure = ssim(ones, zeros, data_range=1., reduction='none')
    assert torch.le(measure, 1).all(), f'SSIM must be <= 1, got {measure}'


def test_ssim_raises_if_tensors_have_different_shapes(x_y_4d_5d, device) -> None:
    y = x_y_4d_5d[1].to(device)
    dims = [[3], [2, 3], [161, 162], [161, 162]]
    if y.dim() == 5:
        dims += [[2, 3]]
    for size in list(itertools.product(*dims)):
        wrong_shape_x = torch.rand(size).to(y)
        if wrong_shape_x.size() == y.size():
            try:
                ssim(wrong_shape_x, y)
            except Exception as e:
                pytest.fail(f"Unexpected error occurred: {e}")
        else:
            with pytest.raises(AssertionError):
                ssim(wrong_shape_x, y)


def test_ssim_raises_if_tensors_have_different_types(y: torch.Tensor) -> None:
    wrong_type_x = list(range(10))
    with pytest.raises(AssertionError):
        ssim(wrong_type_x, y)


def test_ssim_check_available_dimensions() -> None:
    custom_x = torch.rand(256, 256)
    custom_y = torch.rand(256, 256)
    for _ in range(10):
        if custom_x.dim() < 5:
            try:
                ssim(custom_x, custom_y)
            except Exception as e:
                pytest.fail(f"Unexpected error occurred: {e}")
        else:
            with pytest.raises(AssertionError):
                ssim(custom_x, custom_y)
        custom_x.unsqueeze_(0)
        custom_y.unsqueeze_(0)


def test_ssim_check_kernel_size_is_passed(x_y_4d_5d, device: str) -> None:
    x = x_y_4d_5d[0].to(device)
    y = x_y_4d_5d[1].to(device)
    kernel_sizes = list(range(0, 50))
    for kernel_size in kernel_sizes:
        if kernel_size % 2:
            ssim(x, y, kernel_size=kernel_size)
        else:
            with pytest.raises(AssertionError):
                ssim(x, y, kernel_size=kernel_size)


def test_ssim_raises_if_kernel_size_greater_than_image(x_y_4d_5d, device: str) -> None:
    x = x_y_4d_5d[0].to(device)
    y = x_y_4d_5d[1].to(device)
    kernel_size = 11
    wrong_size_x = x[:, :, :kernel_size - 1, :kernel_size - 1]
    wrong_size_y = y[:, :, :kernel_size - 1, :kernel_size - 1]
    with pytest.raises(ValueError):
        ssim(wrong_size_x, wrong_size_y, kernel_size=kernel_size)


def test_ssim_raise_if_wrong_value_is_estimated(test_images: Tuple[torch.Tensor, torch.Tensor],
                                                device: str) -> None:
    for x, y in test_images:
        piq_ssim = ssim(x.to(device), y.to(device), kernel_size=11, kernel_sigma=1.5, data_range=255,
                        reduction='none')
        tf_x = tf.convert_to_tensor(x.permute(0, 2, 3, 1).numpy())
        tf_y = tf.convert_to_tensor(y.permute(0, 2, 3, 1).numpy())
        with tf.device('/CPU'):
            tf_ssim = torch.tensor(tf.image.ssim(tf_x, tf_y, max_val=255).numpy()).to(piq_ssim)

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
    x, y = input_tensors
    x_scaled = (x * data_range).type(torch.uint8)
    y_scaled = (y * data_range).type(torch.uint8)

    measure_scaled = ssim(x_scaled.to(device), y_scaled.to(device), data_range=data_range)
    measure = ssim(
        x_scaled.to(device) / float(data_range),
        y_scaled.to(device) / float(data_range),
        data_range=1.0
    )
    diff = torch.abs(measure_scaled - measure)
    assert diff <= 1e-6, f'Result for same tensor with different data_range should be the same, got {diff}'


def test_ssim_fails_for_incorrect_data_range(x: torch.Tensor, y: torch.Tensor, device: str) -> None:
    # Scale to [0, 255]
    x_scaled = (x * 255).type(torch.uint8)
    y_scaled = (y * 255).type(torch.uint8)
    with pytest.raises(AssertionError):
        ssim(x_scaled.to(device), y_scaled.to(device), data_range=1.0)


# ================== Test class: `SSIMLoss` ==================
def test_ssim_loss_grad(x_y_4d_5d, device: str) -> None:
    x = x_y_4d_5d[0].to(device)
    y = x_y_4d_5d[1].to(device)
    x.requires_grad_(True)
    loss = SSIMLoss(data_range=1.)(x, y).mean()
    loss.backward()
    assert torch.isfinite(x.grad).all(), f'Expected finite gradient values, got {x.grad}'


def test_ssim_loss_symmetry(x_y_4d_5d, device: str) -> None:
    x = x_y_4d_5d[0].to(device)
    y = x_y_4d_5d[1].to(device)
    loss = SSIMLoss()
    loss_value = loss(x, y)
    reverse_loss_value = loss(y, x)
    assert torch.allclose(loss_value, reverse_loss_value), \
        f'Expect: SSIMLoss(a, b) == SSIMLoss(b, a), got {loss_value} != {reverse_loss_value}'


def test_ssim_loss_equality(y: torch.Tensor, device: str) -> None:
    y = y.to(device)
    x = y.clone()
    loss = SSIMLoss()(x, y)
    assert torch.allclose(loss, torch.zeros_like(loss)), \
        f'If equal tensors are passed SSIM loss must be equal to 0 '\
        f'(considering floating point operation error up to 1 * 10^-6), got {loss}'


def test_ssim_loss_is_less_or_equal_to_one(ones_zeros_4d_5d: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    # Create two maximally different tensors.
    ones = ones_zeros_4d_5d[0].to(device)
    zeros = ones_zeros_4d_5d[1].to(device)
    loss = SSIMLoss()(ones, zeros)
    assert (loss <= 1).all(), f'SSIM loss must be <= 1, got {loss}'


def test_ssim_loss_raises_if_tensors_have_different_shapes(x_y_4d_5d,
                                                           device) -> None:
    y = x_y_4d_5d[1].to(device)
    dims = [[3], [2, 3], [161, 162], [161, 162]]
    if y.dim() == 5:
        dims += [[2, 3]]
    for size in list(itertools.product(*dims)):
        wrong_shape_x = torch.rand(size).to(y)
        if wrong_shape_x.size() == y.size():
            SSIMLoss()(wrong_shape_x, y)
        else:
            with pytest.raises(AssertionError):
                SSIMLoss()(wrong_shape_x, y)


def test_ssim_loss_check_available_dimensions() -> None:
    custom_x = torch.rand(256, 256)
    custom_y = torch.rand(256, 256)
    for _ in range(10):
        if custom_x.dim() < 5:
            try:
                SSIMLoss()(custom_x, custom_y)
            except Exception as e:
                pytest.fail(f"Unexpected error occurred: {e}")
        else:
            with pytest.raises(AssertionError):
                SSIMLoss()(custom_x, custom_y)
        custom_x.unsqueeze_(0)
        custom_y.unsqueeze_(0)


def test_ssim_loss_raises_if_tensors_have_different_types(y: torch.Tensor) -> None:
    wrong_type_x = list(range(10))
    with pytest.raises(AssertionError):
        SSIMLoss()(wrong_type_x, y)


def test_ssim_loss_check_kernel_size_is_passed(x: torch.Tensor, y: torch.Tensor) -> None:
    kernel_sizes = list(range(0, 50))
    for kernel_size in kernel_sizes:
        if kernel_size % 2:
            SSIMLoss(kernel_size=kernel_size)(x, y)
        else:
            with pytest.raises(AssertionError):
                SSIMLoss(kernel_size=kernel_size)(x, y)


def test_ssim_loss_raises_if_kernel_size_greater_than_image(x_y_4d_5d, device: str) -> None:
    x = x_y_4d_5d[0].to(device)
    y = x_y_4d_5d[1].to(device)
    kernel_size = 11
    wrong_size_x = x[:, :, :kernel_size - 1, :kernel_size - 1]
    wrong_size_y = y[:, :, :kernel_size - 1, :kernel_size - 1]
    with pytest.raises(ValueError):
        SSIMLoss(kernel_size=kernel_size)(wrong_size_x, wrong_size_y)


def test_ssim_loss_raise_if_wrong_value_is_estimated(test_images: Tuple[torch.Tensor, torch.Tensor],
                                                     device: str) -> None:
    for x, y in test_images:
        ssim_loss = SSIMLoss(kernel_size=11, kernel_sigma=1.5, data_range=255, reduction='mean')(x.to(device),
                                                                                                 y.to(device))
        tf_x = tf.convert_to_tensor(x.permute(0, 2, 3, 1).numpy())
        tf_y = tf.convert_to_tensor(y.permute(0, 2, 3, 1).numpy())
        with tf.device('/CPU'):
            tf_ssim = torch.tensor(tf.image.ssim(tf_x, tf_y, max_val=255).numpy()).mean().to(device)
        match_accuracy = 2e-4 + 1e-8
        assert torch.isclose(ssim_loss, 1. - tf_ssim, rtol=0, atol=match_accuracy), \
            f'The estimated value must be equal to tensorflow provided one' \
            f'(considering floating point operation error up to {match_accuracy}), ' \
            f'got difference {(ssim_loss - 1. + tf_ssim).abs()}'


# ================== Test function: `multi_scale_ssim` ==================
def test_multi_scale_ssim_symmetry(x_y_4d_5d, device: str) -> None:
    x = x_y_4d_5d[0].to(device)
    y = x_y_4d_5d[1].to(device)
    measure = multi_scale_ssim(x, y, data_range=1., reduction='none')
    reverse_measure = multi_scale_ssim(y, x, data_range=1., reduction='none')
    assert torch.allclose(measure, reverse_measure), f'Expect: MS-SSIM(a, b) == MSSSIM(b, a), '\
                                                     f'got {measure} != {reverse_measure}'


def test_multi_scale_ssim_measure_is_one_for_equal_tensors(x: torch.Tensor, device: str) -> None:
    x = x.to(device)
    y = x.clone()
    measure = multi_scale_ssim(y, x, data_range=1.)
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


def test_multi_scale_ssim_raises_if_tensors_have_different_shapes(x_y_4d_5d, device: str) -> None:
    y = x_y_4d_5d[1].to(device)
    dims = [[3], [2, 3], [161, 162], [161, 162]]
    if y.dim() == 5:
        dims += [[2, 3]]
    for size in list(itertools.product(*dims)):
        wrong_shape_x = torch.rand(size).to(y)
        if wrong_shape_x.size() == y.size():
            multi_scale_ssim(wrong_shape_x, y)
        else:
            with pytest.raises(AssertionError):
                multi_scale_ssim(wrong_shape_x, y)
    scale_weights = torch.rand(2, 2)
    with pytest.raises(AssertionError):
        multi_scale_ssim(x, y, scale_weights=scale_weights)


def test_multi_scale_ssim_check_available_dimensions() -> None:
    custom_x = torch.rand(256, 256)
    custom_y = torch.rand(256, 256)
    for _ in range(10):
        if custom_x.dim() < 5:
            try:
                multi_scale_ssim(custom_x, custom_y)
            except Exception as e:
                pytest.fail(f"Unexpected error occurred: {e}")
        else:
            with pytest.raises(AssertionError):
                multi_scale_ssim(custom_x, custom_y)

        custom_x.unsqueeze_(0)
        custom_y.unsqueeze_(0)


def test_multi_scale_ssim_raises_if_tensors_have_different_types(x, y) -> None:
    wrong_type_x = list(range(10))
    with pytest.raises(AssertionError):
        multi_scale_ssim(wrong_type_x, y)
    wrong_type_scale_weights = True
    with pytest.raises(AssertionError):
        multi_scale_ssim(x, y, scale_weights=wrong_type_scale_weights)


def test_multi_scale_ssim_check_kernel_size_is_passed(x, y) -> None:
    kernel_sizes = list(range(0, 13))
    for kernel_size in kernel_sizes:
        if kernel_size % 2:
            multi_scale_ssim(x, y, kernel_size=kernel_size)
        else:
            with pytest.raises(AssertionError):
                multi_scale_ssim(x, y, kernel_size=kernel_size)


def test_ms_ssim_raises_if_kernel_size_greater_than_image(x_y_4d_5d, device: str) -> None:
    x = x_y_4d_5d[0].to(device)
    y = x_y_4d_5d[1].to(device)
    kernel_size = 11
    levels = 5
    min_size = (kernel_size - 1) * 2 ** (levels - 1) + 1
    wrong_size_x = x[:, :, :min_size - 1, :min_size - 1]
    wrong_size_y = y[:, :, :min_size - 1, :min_size - 1]
    with pytest.raises(ValueError):
        multi_scale_ssim(wrong_size_x, wrong_size_y, kernel_size=kernel_size)


def test_multi_scale_ssim_raise_if_wrong_value_is_estimated(test_images: Tuple[torch.Tensor, torch.Tensor],
                                                            scale_weights: List, device: str) -> None:
    for x, y in test_images:
        piq_ms_ssim = multi_scale_ssim(x.to(device), y.to(device), kernel_size=11, kernel_sigma=1.5,
                                       data_range=255, reduction='none', scale_weights=scale_weights)
        tf_x = tf.convert_to_tensor(x.permute(0, 2, 3, 1).numpy())
        tf_y = tf.convert_to_tensor(y.permute(0, 2, 3, 1).numpy())
        with tf.device('/CPU'):
            tf_ms_ssim = torch.tensor(tf.image.ssim_multiscale(tf_x, tf_y, max_val=255,
                                                               power_factors=scale_weights).numpy()).to(device)
        match_accuracy = 1e-5 + 1e-8
        assert torch.allclose(piq_ms_ssim, tf_ms_ssim, rtol=0, atol=match_accuracy), \
            f'The estimated value must be equal to tensorflow provided one' \
            f'(considering floating point operation error up to {match_accuracy}), ' \
            f'got difference {(piq_ms_ssim - tf_ms_ssim).abs()}'


@pytest.mark.parametrize(
    "data_range", [128, 255],
)
def test_multi_scale_ssim_supports_different_data_ranges(x_y_4d_5d, data_range, device: str) -> None:
    x, y = x_y_4d_5d
    x_scaled = (x * data_range).type(torch.uint8)
    y_scaled = (y * data_range).type(torch.uint8)

    measure_scaled = multi_scale_ssim(x_scaled.to(device), y_scaled.to(device), data_range=data_range)
    measure = multi_scale_ssim(
        x_scaled.to(device) / float(data_range),
        y_scaled.to(device) / float(data_range),
        data_range=1.0
    )
    diff = torch.abs(measure_scaled - measure)
    assert (diff <= 1e-6).all(), f'Result for same tensor with different data_range should be the same, got {diff}'


def test_multi_scale_ssim_fails_for_incorrect_data_range(x, y, device: str) -> None:
    # Scale to [0, 255]
    x_scaled = (x * 255).type(torch.uint8)
    y_scaled = (y * 255).type(torch.uint8)
    with pytest.raises(AssertionError):
        multi_scale_ssim(x_scaled.to(device), y_scaled.to(device), data_range=1.0)


# ================== Test class: `MultiScaleSSIMLoss` ==================
def test_multi_scale_ssim_loss_grad(x_y_4d_5d, device: str) -> None:
    x = x_y_4d_5d[0].to(device)
    y = x_y_4d_5d[1].to(device)
    x.requires_grad_()
    loss = MultiScaleSSIMLoss(data_range=1.)(x, y).mean()
    loss.backward()
    assert torch.isfinite(x.grad).all(), f'Expected finite gradient values, got {x.grad}'


def test_multi_scale_ssim_loss_symmetry(x_y_4d_5d, device: str) -> None:
    x = x_y_4d_5d[0].to(device)
    y = x_y_4d_5d[1].to(device)
    loss = MultiScaleSSIMLoss()
    loss_value = loss(x, y)
    reverse_loss_value = loss(y, x)
    assert (loss_value == reverse_loss_value).all(), \
        f'Expect: MS-SSIM(a, b) == MS-SSIM(b, a), got {loss_value} != {reverse_loss_value}'


def test_multi_scale_ssim_loss_equality(y, device: str) -> None:
    y = y.to(device)
    x = y.clone()
    loss = MultiScaleSSIMLoss()(x, y)
    assert (loss.abs() <= 1e-6).all(), f'If equal tensors are passed SSIM loss must be equal to 0 ' \
                                       f'(considering floating point operation error up to 1 * 10^-6), got {loss}'


def test_multi_scale_ssim_loss_is_less_or_equal_to_one(ones_zeros_4d_5d: Tuple[torch.Tensor, torch.Tensor],
                                                       device: str) -> None:
    # Create two maximally different tensors.
    ones = ones_zeros_4d_5d[0].to(device)
    zeros = ones_zeros_4d_5d[1].to(device)
    loss = MultiScaleSSIMLoss()(ones, zeros)
    assert (loss <= 1).all(), f'MS-SSIM loss must be <= 1, got {loss}'


def test_multi_scale_ssim_loss_raises_if_tensors_have_different_shapes(x_y_4d_5d, device: str) -> None:
    y = x_y_4d_5d[1].to(device)
    dims = [[3], [2, 3], [161, 162], [161, 162]]
    if y.dim() == 5:
        dims += [[2, 3]]
    for size in list(itertools.product(*dims)):
        wrong_shape_x = torch.rand(size).to(y)
        if wrong_shape_x.size() == y.size():
            MultiScaleSSIMLoss()(wrong_shape_x, y)
        else:
            with pytest.raises(AssertionError):
                MultiScaleSSIMLoss()(wrong_shape_x, y)

    scale_weights = torch.rand(2, 2)
    with pytest.raises(AssertionError):
        MultiScaleSSIMLoss(scale_weights=scale_weights)(x, y)


def test_multi_scale_ssim_loss_check_available_dimensions() -> None:
    custom_x = torch.rand(256, 256)
    custom_y = torch.rand(256, 256)
    for _ in range(10):
        if custom_x.dim() < 5:
            try:
                MultiScaleSSIMLoss()(custom_x, custom_y)
            except Exception as e:
                pytest.fail(f"Unexpected error occurred: {e}")
        else:
            with pytest.raises(AssertionError):
                MultiScaleSSIMLoss()(custom_x, custom_y)
        custom_x.unsqueeze_(0)
        custom_y.unsqueeze_(0)


def test_multi_scale_ssim_loss_raises_if_tensors_have_different_types(x, y) -> None:
    wrong_type_y = list(range(10))
    with pytest.raises(AssertionError):
        MultiScaleSSIMLoss()(wrong_type_y, y)
    wrong_type_scale_weights = True
    with pytest.raises(AssertionError):
        MultiScaleSSIMLoss(scale_weights=wrong_type_scale_weights)(x, y)


def test_multi_scale_ssim_loss_raises_if_wrong_kernel_size_is_passed(x, y) -> None:
    kernel_sizes = list(range(0, 13))
    for kernel_size in kernel_sizes:
        if kernel_size % 2:
            MultiScaleSSIMLoss(kernel_size=kernel_size)(x, y)
        else:
            with pytest.raises(AssertionError):
                MultiScaleSSIMLoss(kernel_size=kernel_size)(x, y)


def test_ms_ssim_loss_raises_if_kernel_size_greater_than_image(x_y_4d_5d, device: str) -> None:
    x = x_y_4d_5d[0].to(device)
    y = x_y_4d_5d[1].to(device)
    kernel_size = 11
    levels = 5
    min_size = (kernel_size - 1) * 2 ** (levels - 1) + 1
    wrong_size_x = x[:, :, :min_size - 1, :min_size - 1]
    wrong_size_y = y[:, :, :min_size - 1, :min_size - 1]
    with pytest.raises(ValueError):
        MultiScaleSSIMLoss(kernel_size=kernel_size)(wrong_size_x, wrong_size_y)


def test_multi_scale_ssim_loss_raise_if_wrong_value_is_estimated(test_images: List, scale_weights: List,
                                                                 device: str) -> None:
    for x, y in test_images:
        piq_loss = MultiScaleSSIMLoss(kernel_size=11, kernel_sigma=1.5, data_range=255, scale_weights=scale_weights)
        piq_ms_ssim_loss = piq_loss(x.to(device), y.to(device))
        tf_x = tf.convert_to_tensor(x.permute(0, 2, 3, 1).numpy())
        tf_y = tf.convert_to_tensor(y.permute(0, 2, 3, 1).numpy())
        with tf.device('/CPU'):
            tf_ms_ssim = torch.tensor(tf.image.ssim_multiscale(tf_x, tf_y,
                                                               power_factors=scale_weights,
                                                               max_val=255).numpy()).mean().to(device)
        match_accuracy = 1e-5 + 1e-8
        assert torch.isclose(piq_ms_ssim_loss, 1. - tf_ms_ssim, rtol=0, atol=match_accuracy), \
            f'The estimated value must be equal to tensorflow provided one' \
            f'(considering floating point operation error up to {match_accuracy}), ' \
            f'got difference {(piq_ms_ssim_loss - 1. + tf_ms_ssim).abs()}'
