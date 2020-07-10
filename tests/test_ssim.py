import torch
import itertools
import pytest
import tensorflow as tf
import numpy as np
from PIL import Image
from piq import SSIMLoss, MultiScaleSSIMLoss, ssim, multi_scale_ssim


@pytest.fixture(scope='module')
def prediction() -> torch.Tensor:
    return torch.rand(3, 3, 256, 256)


@pytest.fixture(scope='module')
def target() -> torch.Tensor:
    return torch.rand(3, 3, 256, 256)


@pytest.fixture(scope='module')
def prediction_5d() -> torch.Tensor:
    return torch.rand(3, 3, 256, 256, 2)


@pytest.fixture(scope='module')
def target_5d() -> torch.Tensor:
    return torch.rand(3, 3, 256, 256, 2)


# ================== Test function: `ssim` ==================
def test_ssim_symmetry(prediction: torch.Tensor, target: torch.Tensor) -> None:
    measure = ssim(prediction, target, data_range=1., reduction='none')
    reverse_measure = ssim(target, prediction, data_range=1., reduction='none')
    assert torch.allclose(measure, reverse_measure), f'Expect: SSIM(a, b) == SSIM(b, a), got {measure} != {reverse_measure}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test on GPU if there is no GPU.')
def test_ssim_symmetry_cuda(prediction: torch.Tensor, target: torch.Tensor) -> None:
    prediction = prediction.cuda()
    target = target.cuda()
    test_ssim_symmetry(prediction=prediction, target=target)


def test_ssim_symmetry_5d(prediction_5d: torch.Tensor, target_5d: torch.Tensor) -> None:
    test_ssim_symmetry(prediction_5d, target_5d)


def test_ssim_measure_is_one_for_equal_tensors(target: torch.Tensor) -> None:
    prediction = target.clone()
    measure = ssim(prediction, target, data_range=1., reduction='none')
    assert torch.allclose(measure, torch.ones_like(measure)), f'If equal tensors are passed SSIM must be equal to 1 ' \
                                                              f'(considering floating point error up to 1 * 10^-6), '\
                                                              f'got {measure + 1}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test on GPU if there is no GPU.')
def test_ssim_measure_is_one_for_equal_tensors_cuda(target: torch.Tensor) -> None:
    target = target.cuda()
    test_ssim_measure_is_one_for_equal_tensors(target=target)


def test_ssim_measure_is_less_or_equal_to_one() -> None:
    # Create two maximally different tensors.
    ones = torch.ones((3, 3, 256, 256))
    zeros = torch.zeros((3, 3, 256, 256))
    measure = ssim(ones, zeros, data_range=1., reduction='none')
    assert (measure <= 1).all(), f'SSIM must be <= 1, got {measure}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test on GPU if there is no GPU.')
def test_ssim_measure_is_less_or_equal_to_one_cuda() -> None:
    ones = torch.ones((3, 3, 256, 256)).cuda()
    zeros = torch.zeros((3, 3, 256, 256)).cuda()
    measure = ssim(ones, zeros, data_range=1., reduction='none')
    assert (measure <= 1).all(), f'SSIM must be <= 1, got {measure}'


def test_ssim_measure_is_less_or_equal_to_one_5d() -> None:
    ones = torch.ones((3, 3, 256, 256, 2))
    zeros = torch.zeros((3, 3, 256, 256, 2))
    measure = ssim(ones, zeros, data_range=1., reduction='none')
    assert (measure <= 1).all(), f'SSIM must be <= 1, got {measure}'


def test_ssim_raises_if_tensors_have_different_dimensions() -> None:
    custom_prediction = torch.rand(256, 256)
    with pytest.raises(AssertionError):
        ssim(custom_prediction, custom_prediction.unsqueeze(0))


def test_ssim_raises_if_tensors_have_different_shapes(target: torch.Tensor) -> None:
    dims = [[3], [2, 3], [255, 256], [255, 256]]
    for b, c, h, w in list(itertools.product(*dims)):
        wrong_shape_prediction = torch.rand(b, c, h, w)
        if wrong_shape_prediction.size() == target.size():
            try:
                ssim(wrong_shape_prediction, target)
            except Exception as e:
                pytest.fail(f"Unexpected error occurred: {e}")
        else:
            with pytest.raises(AssertionError):
                ssim(wrong_shape_prediction, target)


def test_ssim_raises_if_tensors_have_different_shapes_5d(target_5d: torch.Tensor) -> None:
    dims = [[3], [2, 3], [255, 256], [255, 256], [2, 3]]
    for b, c, h, w, d in list(itertools.product(*dims)):
        wrong_shape_prediction = torch.rand(b, c, h, w, d)
        if wrong_shape_prediction.size() == target_5d.size():
            try:
                ssim(wrong_shape_prediction, target_5d)
            except Exception as e:
                pytest.fail(f"Unexpected error occurred: {e}")
        else:
            with pytest.raises(AssertionError):
                ssim(wrong_shape_prediction, target_5d)


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


def test_ssim_raises_if_wrong_kernel_size_is_passed(prediction: torch.Tensor, target: torch.Tensor) -> None:
    wrong_kernel_sizes = list(range(0, 50, 2))
    for kernel_size in wrong_kernel_sizes:
        with pytest.raises(AssertionError):
            ssim(prediction, target, kernel_size=kernel_size)


def test_ssim_raises_if_kernel_size_greater_than_image() -> None:
    right_kernel_sizes = list(range(1, 52, 2))
    for kernel_size in right_kernel_sizes:
        wrong_size_prediction = torch.rand(3, 3, kernel_size - 1, kernel_size - 1)
        wrong_size_target = torch.rand(3, 3, kernel_size - 1, kernel_size - 1)
        with pytest.raises(ValueError):
            ssim(wrong_size_prediction, wrong_size_target, kernel_size=kernel_size)


def test_ssim_raise_if_wrong_value_is_estimated() -> None:
    prediction_grey = torch.tensor(np.array(Image.open('tests/assets/goldhill_jpeg.gif'))).unsqueeze(0).unsqueeze(0)
    target_grey = torch.tensor(np.array(Image.open('tests/assets/goldhill.gif'))).unsqueeze(0).unsqueeze(0)
    prediction_rgb = torch.tensor(np.array(Image.open('tests/assets/I01.BMP'))).permute(2, 0, 1).unsqueeze(0)
    target_rgb = torch.tensor(np.array(Image.open('tests/assets/i01_01_5.bmp'))).permute(2, 0, 1).unsqueeze(0)
    for prediction, target in [(prediction_grey, target_grey), (prediction_rgb, target_rgb)]:
        piq_ssim = ssim(prediction, target, kernel_size=11, kernel_sigma=1.5, data_range=255, reduction='none')
        tf_prediction = tf.convert_to_tensor(prediction.permute(0, 2, 3, 1).numpy())
        tf_target = tf.convert_to_tensor(target.permute(0, 2, 3, 1).numpy())
        tf_ssim = torch.tensor(tf.image.ssim(tf_prediction, tf_target, max_val=255).numpy())
        match_accuracy = 2e-5 + 1e-8
        assert torch.allclose(piq_ssim, tf_ssim, rtol=0,  atol=match_accuracy), \
            f'The estimated value must be equal to tensorflow provided one' \
            f'(considering floating point operation error up to {match_accuracy}), ' \
            f'got difference {(piq_ssim - tf_ssim).abs()}'


# ================== Test class: `SSIMLoss` ==================
def test_ssim_loss_grad(prediction: torch.Tensor, target: torch.Tensor) -> None:
    prediction.requires_grad_()
    loss = SSIMLoss(data_range=1.)(prediction, target)
    loss.backward()
    assert prediction.grad is not None, f'Expected finite gradient values'


def test_ssim_loss_symmetry(prediction: torch.Tensor, target: torch.Tensor) -> None:
    loss = SSIMLoss()
    loss_value = loss(prediction, target.detach())
    reverse_loss_value = loss(target, prediction.detach())
    assert torch.allclose(loss_value, reverse_loss_value), \
        f'Expect: SSIMLoss(a, b) == SSIMLoss(b, a), got {loss_value} != {reverse_loss_value}'



@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test on GPU if there is no GPU.')
def test_ssim_loss_symmetry_cuda(prediction: torch.Tensor, target: torch.Tensor) -> None:
    prediction = prediction.cuda()
    target = target.cuda()
    test_ssim_loss_symmetry(prediction=prediction, target=target)


def test_ssim_loss_symmetry_5d(prediction_5d: torch.Tensor, target_5d: torch.Tensor) -> None:
    test_ssim_loss_symmetry(prediction_5d, target_5d)


def test_ssim_loss_equality(target: torch.Tensor) -> None:
    prediction = target.clone()
    loss = SSIMLoss()(prediction, target)
    assert torch.allclose(loss, torch.zeros_like(loss)), \
        f'If equal tensors are passed SSIM loss must be equal to 0 '\
        f'(considering floating point operation error up to 1 * 10^-6), got {loss}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test on GPU if there is no GPU.')
def test_ssim_loss_equality_cuda(target: torch.Tensor) -> None:
    target = target.cuda()
    test_ssim_loss_equality(target=target)


def test_ssim_loss_is_less_or_equal_to_one() -> None:
    # Create two maximally different tensors.
    ones = torch.ones((3, 3, 256, 256))
    zeros = torch.zeros((3, 3, 256, 256))
    loss = SSIMLoss()(ones, zeros)
    assert (loss <= 1).all(), f'SSIM loss must be <= 1, got {loss}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test on GPU if there is no GPU.')
def test_ssim_loss_is_less_or_equal_to_one_cuda() -> None:
    ones = torch.ones((3, 3, 256, 256)).cuda()
    zeros = torch.zeros((3, 3, 256, 256)).cuda()
    loss = SSIMLoss()(ones, zeros)
    assert (loss <= 1).all(), f'SSIM loss must be <= 1, got {loss}'


def test_ssim_loss_is_less_or_equal_to_one_5d() -> None:
    # Create two maximally different tensors.
    ones = torch.ones((3, 3, 256, 256, 2))
    zeros = torch.zeros((3, 3, 256, 256, 2))
    loss = SSIMLoss()(ones, zeros)
    assert (loss <= 1).all(), f'SSIM loss must be <= 1, got {loss}'


def test_ssim_loss_raises_if_tensors_have_different_dimensions() -> None:
    custom_prediction = torch.rand(256, 256)
    with pytest.raises(AssertionError):
        SSIMLoss()(custom_prediction, custom_prediction.unsqueeze(0))


def test_ssim_loss_raises_if_tensors_have_different_shapes(target: torch.Tensor) -> None:
    dims = [[3], [2, 3], [255, 256], [255, 256]]
    for b, c, h, w in list(itertools.product(*dims)):
        wrong_shape_prediction = torch.rand(b, c, h, w)
        if wrong_shape_prediction.size() == target.size():
            try:
                SSIMLoss()(wrong_shape_prediction, target)
            except Exception as e:
                pytest.fail(f"Unexpected error occurred: {e}")
        else:
            with pytest.raises(AssertionError):
                SSIMLoss()(wrong_shape_prediction, target)


def test_ssim_loss_raises_if_tensors_have_different_shapes_5d(target_5d: torch.Tensor) -> None:
    dims = [[3], [2, 3], [255, 256], [255, 256], [2, 3]]
    for b, c, h, w, d in list(itertools.product(*dims)):
        wrong_shape_prediction = torch.rand(b, c, h, w, d)
        if wrong_shape_prediction.size() == target_5d.size():
            try:
                SSIMLoss()(wrong_shape_prediction, target_5d)
            except Exception as e:
                pytest.fail(f"Unexpected error occurred: {e}")
        else:
            with pytest.raises(AssertionError):
                SSIMLoss()(wrong_shape_prediction, target_5d)


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


def test_ssim_loss_raises_if_wrong_kernel_size_is_passed(prediction: torch.Tensor, target: torch.Tensor) -> None:
    wrong_kernel_sizes = list(range(0, 50, 2))
    for kernel_size in wrong_kernel_sizes:
        with pytest.raises(AssertionError):
            SSIMLoss(kernel_size=kernel_size)(prediction, target)


def test_ssim_loss_raises_if_kernel_size_greater_than_image() -> None:
    right_kernel_sizes = list(range(1, 52, 2))
    for kernel_size in right_kernel_sizes:
        wrong_size_prediction = torch.rand(3, 3, kernel_size - 1, kernel_size - 1)
        wrong_size_target = torch.rand(3, 3, kernel_size - 1, kernel_size - 1)
        with pytest.raises(ValueError):
            SSIMLoss(kernel_size=kernel_size)(wrong_size_prediction, wrong_size_target)


def test_ssim_loss_raise_if_wrong_value_is_estimated() -> None:
    prediction_grey = torch.tensor(np.array(Image.open('tests/assets/goldhill_jpeg.gif'))).unsqueeze(0).unsqueeze(0)
    target_grey = torch.tensor(np.array(Image.open('tests/assets/goldhill.gif'))).unsqueeze(0).unsqueeze(0)
    prediction_rgb = torch.tensor(np.array(Image.open('tests/assets/I01.BMP'))).permute(2, 0, 1).unsqueeze(0)
    target_rgb = torch.tensor(np.array(Image.open('tests/assets/i01_01_5.bmp'))).permute(2, 0, 1).unsqueeze(0)
    for prediction, target in [(prediction_grey, target_grey), (prediction_rgb, target_rgb)]:
        ssim_loss = SSIMLoss(kernel_size=11, kernel_sigma=1.5, data_range=255, reduction='mean')(prediction, target)
        tf_prediction = tf.convert_to_tensor(prediction.permute(0, 2, 3, 1).numpy())
        tf_target = tf.convert_to_tensor(target.permute(0, 2, 3, 1).numpy())
        tf_ssim = torch.tensor(tf.image.ssim(tf_prediction, tf_target, max_val=255).numpy()).mean()
        match_accuracy = 2e-5 + 1e-8
        assert torch.isclose(ssim_loss, 1. - tf_ssim, rtol=0, atol=match_accuracy), \
            f'The estimated value must be equal to tensorflow provided one' \
            f'(considering floating point operation error up to {match_accuracy}), ' \
            f'got difference {(ssim_loss - 1. + tf_ssim).abs()}'


# ================== Test function: `multi_scale_ssim` ==================
def test_multi_scale_ssim_symmetry(prediction: torch.Tensor, target: torch.Tensor) -> None:
    measure = multi_scale_ssim(prediction, target, data_range=1., reduction='none')
    reverse_measure = multi_scale_ssim(target, prediction, data_range=1., reduction='none')
    assert torch.allclose(measure, reverse_measure), f'Expect: MS-SSIM(a, b) == MSSSIM(b, a), '\
                                                     f'got {measure} != {reverse_measure}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test on GPU if there is no GPU.')
def test_multi_scale_ssim_symmetry_cuda(prediction: torch.Tensor, target: torch.Tensor) -> None:
    prediction = prediction.cuda()
    target = target.cuda()
    test_multi_scale_ssim_loss_symmetry(prediction=prediction, target=target)


def test_multi_scale_ssim_symmetry_5d(prediction_5d: torch.Tensor, target_5d: torch.Tensor) -> None:
    measure = multi_scale_ssim(prediction_5d, target_5d, data_range=1., k2=.4)
    reverse_measure = multi_scale_ssim(target_5d, prediction_5d, data_range=1., k2=.4)
    assert torch.allclose(measure, reverse_measure), f'Expect: MS-SSIM(a, b) == MS-SSIM(b, a), '\
                                                     f'got {measure} != {reverse_measure}'


def test_multi_scale_ssim_measure_is_one_for_equal_tensors(target: torch.Tensor) -> None:
    prediction = target.clone()
    measure = multi_scale_ssim(prediction, target, data_range=1.)
    assert torch.allclose(measure, torch.ones_like(measure)), \
        f'If equal tensors are passed MS-SSIM must be equal to 1 ' \
        f'(considering floating point operation error up to 1 * 10^-6), got {measure + 1}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test on GPU if there is no GPU.')
def test_multi_scale_ssim_measure_is_one_for_equal_tensors_cuda(target: torch.Tensor) -> None:
    target = target.cuda()
    test_multi_scale_ssim_measure_is_one_for_equal_tensors(target=target)


def test_multi_scale_ssim_measure_is_less_or_equal_to_one() -> None:
    # Create two maximally different tensors.
    ones = torch.ones((3, 3, 256, 256))
    zeros = torch.zeros((3, 3, 256, 256))
    measure = multi_scale_ssim(ones, zeros, data_range=1.)
    assert (measure <= 1).all(), f'MS-SSIM must be <= 1, got {measure}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test on GPU if there is no GPU.')
def test_multi_scale_ssim_measure_is_less_or_equal_to_one_cuda() -> None:
    ones = torch.ones((3, 3, 256, 256)).cuda()
    zeros = torch.zeros((3, 3, 256, 256)).cuda()
    measure = multi_scale_ssim(ones, zeros, data_range=1.)
    assert (measure <= 1).all(), f'MS-SSIM must be <= 1, got {measure}'


def test_multi_scale_ssim_measure_is_less_or_equal_to_one_5d() -> None:
    # Create two maximally different tensors.
    ones = torch.ones((3, 3, 256, 256, 2))
    zeros = torch.zeros((3, 3, 256, 256, 2))
    measure = multi_scale_ssim(ones, zeros, data_range=1.)
    assert (measure <= 1).all(), f'MS-SSIM must be <= 1, got {measure}'


def test_multi_scale_ssim_raises_if_tensors_have_different_dimensions() -> None:
    custom_prediction = torch.rand(256, 256)
    with pytest.raises(AssertionError):
        multi_scale_ssim(custom_prediction, custom_prediction.unsqueeze(0))


def test_multi_scale_ssim_raises_if_tensors_have_different_shapes(prediction: torch.Tensor,
                                                                  target: torch.Tensor) -> None:
    dims = [[3], [2, 3], [255, 256], [255, 256]]
    for b, c, h, w in list(itertools.product(*dims)):
        wrong_shape_prediction = torch.rand(b, c, h, w)
        if wrong_shape_prediction.size() == target.size():
            try:
                multi_scale_ssim(wrong_shape_prediction, target)
            except Exception as e:
                pytest.fail(f"Unexpected error occurred: {e}")
        else:
            with pytest.raises(AssertionError):
                multi_scale_ssim(wrong_shape_prediction, target)
    scale_weights = torch.rand(2, 2)
    with pytest.raises(AssertionError):
        multi_scale_ssim(prediction, target, scale_weights=scale_weights)


def test_multi_scale_ssim_raises_if_tensors_have_different_shapes_5d(prediction_5d: torch.Tensor,
                                                                     target_5d: torch.Tensor) -> None:
    dims = [[3], [2, 3], [255, 256], [255, 256], [2, 3]]
    for b, c, h, w, d in list(itertools.product(*dims)):
        wrong_shape_prediction = torch.rand(b, c, h, w, d)
        if wrong_shape_prediction.size() == target_5d.size():
            try:
                multi_scale_ssim(wrong_shape_prediction, target_5d)
            except Exception as e:
                pytest.fail(f"Unexpected error occurred: {e}")
        else:
            with pytest.raises(AssertionError):
                multi_scale_ssim(wrong_shape_prediction, target_5d)
    scale_weights = torch.rand(2, 2)
    with pytest.raises(AssertionError):
        multi_scale_ssim(prediction_5d, target_5d, scale_weights=scale_weights)


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


def test_multi_scale_ssim_raises_if_wrong_kernel_size_is_passed(prediction: torch.Tensor, target: torch.Tensor) -> None:
    wrong_kernel_sizes = list(range(0, 50, 2))
    for kernel_size in wrong_kernel_sizes:
        with pytest.raises(AssertionError):
            multi_scale_ssim(prediction, target, kernel_size=kernel_size)


def test_multi_scale_ssim_raises_if_kernel_size_greater_than_image() -> None:
    right_kernel_sizes = list(range(1, 52, 2))
    for kernel_size in right_kernel_sizes:
        wrong_size_prediction = torch.rand(3, 3, kernel_size - 1, kernel_size - 1)
        wrong_size_target = torch.rand(3, 3, kernel_size - 1, kernel_size - 1)
        with pytest.raises(ValueError):
            multi_scale_ssim(wrong_size_prediction, wrong_size_target, kernel_size=kernel_size)


def test_multi_scale_ssim_raise_if_wrong_value_is_estimated() -> None:
    prediction_grey = torch.tensor(np.array(Image.open('tests/assets/goldhill_jpeg.gif'))).unsqueeze(0).unsqueeze(0)
    target_grey = torch.tensor(np.array(Image.open('tests/assets/goldhill.gif'))).unsqueeze(0).unsqueeze(0)
    prediction_rgb = torch.tensor(np.array(Image.open('tests/assets/I01.BMP'))).permute(2, 0, 1).unsqueeze(0)
    target_rgb = torch.tensor(np.array(Image.open('tests/assets/i01_01_5.bmp'))).permute(2, 0, 1).unsqueeze(0)
    for prediction, target in [(prediction_grey, target_grey), (prediction_rgb, target_rgb)]:
        piq_ms_ssim = multi_scale_ssim(prediction, target, kernel_size=11, kernel_sigma=1.5,
                                       data_range=255, reduction='none')
        tf_prediction = tf.convert_to_tensor(prediction.permute(0, 2, 3, 1).numpy())
        tf_target = tf.convert_to_tensor(target.permute(0, 2, 3, 1).numpy())
        tf_ms_ssim = torch.tensor(tf.image.ssim_multiscale(tf_prediction, tf_target, max_val=255).numpy())
        number_of_weights = 5.
        match_accuracy = number_of_weights * 1e-5 + 1e-8
        assert torch.allclose(piq_ms_ssim, tf_ms_ssim, rtol=0, atol=match_accuracy), \
            f'The estimated value must be equal to tensorflow provided one' \
            f'(considering floating point operation error up to {match_accuracy}), ' \
            f'got difference {(piq_ms_ssim - tf_ms_ssim).abs()}'


def test_multi_scale_ssim_raise_if_wrong_value_is_estimated_custom_weights() -> None:
    scale_weights = [0.0448, 0.2856, 0.3001]
    prediction_grey = torch.tensor(np.array(Image.open('tests/assets/goldhill_jpeg.gif'))).unsqueeze(0).unsqueeze(0)
    target_grey = torch.tensor(np.array(Image.open('tests/assets/goldhill.gif'))).unsqueeze(0).unsqueeze(0)
    prediction_rgb = torch.tensor(np.array(Image.open('tests/assets/I01.BMP'))).permute(2, 0, 1).unsqueeze(0)
    target_rgb = torch.tensor(np.array(Image.open('tests/assets/i01_01_5.bmp'))).permute(2, 0, 1).unsqueeze(0)
    for prediction, target in [(prediction_grey, target_grey), (prediction_rgb, target_rgb)]:
        piq_ms_ssim = multi_scale_ssim(prediction, target, kernel_size=11, kernel_sigma=1.5,
                                       data_range=255, reduction='none', scale_weights=scale_weights)
        tf_prediction = tf.convert_to_tensor(prediction.permute(0, 2, 3, 1).numpy())
        tf_target = tf.convert_to_tensor(target.permute(0, 2, 3, 1).numpy())
        tf_ms_ssim = torch.tensor(tf.image.ssim_multiscale(tf_prediction, tf_target, max_val=255,
                                                           power_factors=scale_weights).numpy())
        number_of_weights = len(scale_weights)
        match_accuracy = number_of_weights * 1e-5 + 1e-8
        assert torch.allclose(piq_ms_ssim, tf_ms_ssim, rtol=0, atol=match_accuracy), \
            f'The estimated value must be equal to tensorflow provided one' \
            f'(considering floating point operation error up to {match_accuracy}), ' \
            f'got difference {(piq_ms_ssim - tf_ms_ssim).abs()}'


# ================== Test class: `MultiScaleSSIMLoss` ==================
def test_multi_scale_ssim_loss_grad(prediction: torch.Tensor, target: torch.Tensor) -> None:
    prediction.requires_grad_()
    loss = MultiScaleSSIMLoss(data_range=1.)(prediction, target)
    loss.backward()
    assert prediction.grad is not None, f'Expected finite gradient values'


def test_multi_scale_ssim_loss_symmetry(prediction: torch.Tensor, target: torch.Tensor) -> None:
    loss = MultiScaleSSIMLoss()
    loss_value = loss(prediction, target)
    reverse_loss_value = loss(target, prediction)
    assert (loss_value == reverse_loss_value).all(), \
        f'Expect: MS-SSIM(a, b) == MS-SSIM(b, a), got {loss_value} != {reverse_loss_value}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test on GPU if there is no GPU.')
def test_multi_scale_ssim_loss_symmetry_cuda(prediction: torch.Tensor, target: torch.Tensor) -> None:
    prediction = prediction.cuda()
    target = target.cuda()
    test_multi_scale_ssim_loss_symmetry(prediction=prediction, target=target)


def test_multi_scale_ssim_loss_symmetry_5d(prediction_5d: torch.Tensor, target_5d: torch.Tensor) -> None:
    loss = MultiScaleSSIMLoss(k2=.4)
    loss_value = loss(prediction_5d, target_5d)
    reverse_loss_value = loss(target_5d, prediction_5d)
    assert (loss_value == reverse_loss_value).all(), \
        f'Expect: MS-SSIM(a, b) == MS-SSIM(b, a), got {loss_value} != {reverse_loss_value}'


def test_multi_scale_ssim_loss_equality(target: torch.Tensor) -> None:
    prediction = target.clone()
    loss = MultiScaleSSIMLoss()(prediction, target)
    assert (loss.abs() <= 1e-6).all(), f'If equal tensors are passed SSIM loss must be equal to 0 ' \
                                       f'(considering floating point operation error up to 1 * 10^-6), got {loss}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test on GPU if there is no GPU.')
def test_multi_scale_ssim_loss_equality_cuda(target: torch.Tensor) -> None:
    target = target.cuda()
    test_multi_scale_ssim_loss_equality(target=target)


def test_multi_scale_ssim_loss_is_less_or_equal_to_one() -> None:
    # Create two maximally different tensors.
    ones = torch.ones((3, 3, 256, 256))
    zeros = torch.zeros((3, 3, 256, 256))
    loss = MultiScaleSSIMLoss()(ones, zeros)
    assert loss <= 1, f'MS-SSIM loss must be <= 1, got {loss}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test on GPU if there is no GPU.')
def test_multi_scale_ssim_loss_is_less_or_equal_to_one_cuda() -> None:
    ones = torch.ones((3, 3, 256, 256)).cuda()
    zeros = torch.zeros((3, 3, 256, 256)).cuda()
    loss = MultiScaleSSIMLoss()(ones, zeros)
    assert loss <= 1, f'MS-SSIM loss must be <= 1, got {loss}'


def test_multi_scale_ssim_loss_is_less_or_equal_to_one_5d() -> None:
    # Create two maximally different tensors.
    ones = torch.ones((3, 3, 256, 256, 2))
    zeros = torch.zeros((3, 3, 256, 256, 2))
    loss = MultiScaleSSIMLoss()(ones, zeros)
    assert (loss <= 1).all(), f'MS-SSIM loss must be <= 1, got {loss}'


def test_multi_scale_ssim_loss_raises_if_tensors_have_different_dimensions() -> None:
    custom_prediction = torch.rand(256, 256)
    with pytest.raises(AssertionError):
        MultiScaleSSIMLoss()(custom_prediction, custom_prediction.unsqueeze(0))


def test_multi_scale_ssim_loss_raises_if_tensors_have_different_shapes(prediction: torch.Tensor,
                                                                       target: torch.Tensor) -> None:
    dims = [[3], [2, 3], [255, 256], [255, 256]]
    for b, c, h, w in list(itertools.product(*dims)):
        wrong_shape_prediction = torch.rand(b, c, h, w)
        if wrong_shape_prediction.size() == target.size():
            try:
                MultiScaleSSIMLoss()(wrong_shape_prediction, target)
            except Exception as e:
                pytest.fail(f"Unexpected error occurred: {e}")
        else:
            with pytest.raises(AssertionError):
                MultiScaleSSIMLoss()(wrong_shape_prediction, target)
    scale_weights = torch.rand(2, 2)
    with pytest.raises(AssertionError):
        MultiScaleSSIMLoss(scale_weights=scale_weights)(prediction, target)


def test_multi_scale_ssim_loss_raises_if_tensors_have_different_shapes_5d(prediction_5d: torch.Tensor,
                                                                          target_5d: torch.Tensor) -> None:
    dims = [[3], [2, 3], [255, 256], [255, 256], [2, 3]]
    for b, c, h, w, d in list(itertools.product(*dims)):
        wrong_shape_prediction = torch.rand(b, c, h, w, d)
        if wrong_shape_prediction.size() == target_5d.size():
            try:
                MultiScaleSSIMLoss()(wrong_shape_prediction, target_5d)
            except Exception as e:
                pytest.fail(f"Unexpected error occurred: {e}")
        else:
            with pytest.raises(AssertionError):
                MultiScaleSSIMLoss()(wrong_shape_prediction, target_5d)
    scale_weights = torch.rand(2, 2)
    with pytest.raises(AssertionError):
        MultiScaleSSIMLoss(scale_weights=scale_weights)(prediction_5d, target_5d)


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
    wrong_kernel_sizes = list(range(0, 50, 2))
    for kernel_size in wrong_kernel_sizes:
        with pytest.raises(AssertionError):
            MultiScaleSSIMLoss(kernel_size=kernel_size)(prediction, target)


def test_multi_scale_ssim_loss_raises_if_kernel_size_greater_than_image() -> None:
    right_kernel_sizes = list(range(1, 52, 2))
    for kernel_size in right_kernel_sizes:
        wrong_size_prediction = torch.rand(3, 3, kernel_size - 1, kernel_size - 1)
        wrong_size_target = torch.rand(3, 3, kernel_size - 1, kernel_size - 1)
        with pytest.raises(ValueError):
            MultiScaleSSIMLoss(kernel_size=kernel_size)(wrong_size_prediction, wrong_size_target)


def test_multi_scale_ssim_loss_raise_if_wrong_value_is_estimated() -> None:
    prediction_grey = torch.tensor(np.array(Image.open('tests/assets/goldhill_jpeg.gif'))).unsqueeze(0).unsqueeze(0)
    target_grey = torch.tensor(np.array(Image.open('tests/assets/goldhill.gif'))).unsqueeze(0).unsqueeze(0)
    prediction_rgb = torch.tensor(np.array(Image.open('tests/assets/I01.BMP'))).permute(2, 0, 1).unsqueeze(0)
    target_rgb = torch.tensor(np.array(Image.open('tests/assets/i01_01_5.bmp'))).permute(2, 0, 1).unsqueeze(0)
    for prediction, target in [(prediction_grey, target_grey), (prediction_rgb, target_rgb)]:
        piq_ms_ssim_loss = MultiScaleSSIMLoss(kernel_size=11, kernel_sigma=1.5,
                                              data_range=255)(prediction, target)
        tf_prediction = tf.convert_to_tensor(prediction.permute(0, 2, 3, 1).numpy())
        tf_target = tf.convert_to_tensor(target.permute(0, 2, 3, 1).numpy())
        tf_ms_ssim = torch.tensor(tf.image.ssim_multiscale(tf_prediction, tf_target, max_val=255).numpy()).mean()
        number_of_weights = 5.
        match_accuracy = number_of_weights * 1e-5 + 1e-8
        assert torch.isclose(piq_ms_ssim_loss, 1. - tf_ms_ssim, rtol=0, atol=match_accuracy), \
            f'The estimated value must be equal to tensorflow provided one' \
            f'(considering floating point operation error up to {match_accuracy}), ' \
            f'got difference {(piq_ms_ssim_loss - 1. + tf_ms_ssim).abs()}'


def test_multi_scale_ssim_loss_raise_if_wrong_value_is_estimated_custom_weights() -> None:
    scale_weights = [0.0448, 0.2856, 0.3001]
    prediction_grey = torch.tensor(np.array(Image.open('tests/assets/goldhill_jpeg.gif'))).unsqueeze(0).unsqueeze(0)
    target_grey = torch.tensor(np.array(Image.open('tests/assets/goldhill.gif'))).unsqueeze(0).unsqueeze(0)
    prediction_rgb = torch.tensor(np.array(Image.open('tests/assets/I01.BMP'))).permute(2, 0, 1).unsqueeze(0)
    target_rgb = torch.tensor(np.array(Image.open('tests/assets/i01_01_5.bmp'))).permute(2, 0, 1).unsqueeze(0)
    for prediction, target in [(prediction_grey, target_grey), (prediction_rgb, target_rgb)]:
        piq_ms_ssim_loss = MultiScaleSSIMLoss(kernel_size=11, kernel_sigma=1.5,
                                              data_range=255, scale_weights=scale_weights)(prediction, target)
        tf_prediction = tf.convert_to_tensor(prediction.permute(0, 2, 3, 1).numpy())
        tf_target = tf.convert_to_tensor(target.permute(0, 2, 3, 1).numpy())
        tf_ms_ssim = torch.tensor(tf.image.ssim_multiscale(tf_prediction, tf_target, max_val=255,
                                                           power_factors=scale_weights).numpy()).mean()
        number_of_weights = len(scale_weights)
        match_accuracy = number_of_weights * 1e-5 + 1e-8
        assert torch.isclose(piq_ms_ssim_loss, 1. - tf_ms_ssim, rtol=0, atol=match_accuracy), \
            f'The estimated value must be equal to tensorflow provided one' \
            f'(considering floating point operation error up to {match_accuracy}), ' \
            f'got difference {(piq_ms_ssim_loss - 1. + tf_ms_ssim).abs()}'
