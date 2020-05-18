import torch
import itertools
import pytest
import tensorflow as tf

from photosynthesis_metrics import SSIMLoss, MultiScaleSSIMLoss, ssim, multi_scale_ssim


@pytest.fixture(scope='module')
def prediction() -> torch.Tensor:
    return torch.rand(3, 3, 256, 256)


@pytest.fixture(scope='module')
def target() -> torch.Tensor:
    return torch.rand(3, 3, 256, 256)


# ================== Test function: `ssim` ==================
def test_ssim_symmetry(prediction: torch.Tensor, target: torch.Tensor) -> None:
    measure = ssim(prediction, target, data_range=1.)
    reverse_measure = ssim(target, prediction, data_range=1.)
    assert measure == reverse_measure, f'Expect: SSIM(a, b) == SSIM(b, a), got {measure} != {reverse_measure}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test on GPU if there is no GPU.')
def test_ssim_symmetry_cuda(prediction: torch.Tensor, target: torch.Tensor) -> None:
    prediction = prediction.cuda()
    target = target.cuda()
    test_ssim_symmetry(prediction=prediction, target=target)


def test_ssim_measure_is_zero_for_equal_tensors(target: torch.Tensor) -> None:
    prediction = target.clone()
    measure = ssim(prediction, target, data_range=1.)
    measure -= 1.
    assert measure.sum() <= 1e-6, f'If equal tensors are passed SSIM must be equal to 0 ' \
                                  f'(considering floating point operation error up to 1 * 10^-6), got {measure}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test on GPU if there is no GPU.')
def test_ssim_measure_is_zero_for_equal_tensors_cuda(target: torch.Tensor) -> None:
    target = target.cuda()
    test_ssim_measure_is_zero_for_equal_tensors(target=target)


def test_ssim_measure_is_less_or_equal_to_one() -> None:
    # Create two maximally different tensors.
    ones = torch.ones((3, 3, 256, 256))
    zeros = torch.zeros((3, 3, 256, 256))
    measure = ssim(ones, zeros, data_range=1.)
    assert measure <= 1, f'SSIM must be <= 1, got {measure}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test on GPU if there is no GPU.')
def test_ssim_measure_is_less_or_equal_to_one_cuda() -> None:
    ones = torch.ones((3, 3, 256, 256)).cuda()
    zeros = torch.zeros((3, 3, 256, 256)).cuda()
    measure = ssim(ones, zeros, data_range=1.)
    assert measure <= 1, f'SSIM must be <= 1, got {measure}'


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


def test_ssim_raises_if_tensors_have_different_types(target: torch.Tensor) -> None:
    wrong_type_prediction = list(range(10))
    with pytest.raises(AssertionError):
        ssim(wrong_type_prediction, target)


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


def test_ssim_raise_if_wrong_value_is_estimated(prediction: torch.Tensor, target: torch.Tensor) -> None:
    photosynthesis_ssim = ssim(prediction, target, kernel_size=11, kernel_sigma=1.5, data_range=1., size_average=False)
    tf_prediction = tf.convert_to_tensor(prediction.permute(0, 2, 3, 1).numpy())
    tf_target = tf.convert_to_tensor(target.permute(0, 2, 3, 1).numpy())
    tf_ssim = torch.tensor(tf.image.ssim(tf_prediction, tf_target, max_val=1.).numpy())
    assert torch.isclose(photosynthesis_ssim, tf_ssim, atol=1e-6).all(), \
        f'The estimated value must be equal to tensorflow provided one' \
        f'(considering floating point operation error up to 1 * 10^-6), ' \
        f'got difference {photosynthesis_ssim - tf_ssim}'


# ================== Test class: `SSIMLoss` ==================
def test_ssim_loss_symmetry(prediction: torch.Tensor, target: torch.Tensor) -> None:
    loss = SSIMLoss()
    loss_value = loss(prediction, target)
    reverse_loss_value = loss(target, prediction)
    assert loss_value == reverse_loss_value, \
        f'Expect: SSIM(a, b) == SSIM(b, a), got {loss_value} != {reverse_loss_value}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test on GPU if there is no GPU.')
def test_ssim_loss_symmetry_cuda(prediction: torch.Tensor, target: torch.Tensor) -> None:
    prediction = prediction.cuda()
    target = target.cuda()
    test_ssim_loss_symmetry(prediction=prediction, target=target)


def test_ssim_loss_equality(target: torch.Tensor) -> None:
    prediction = target.clone()
    loss = SSIMLoss()(prediction, target)
    loss -= 1.
    assert loss.sum() <= 1e-6, f'If equal tensors are passed SSIM loss must be equal to 0 ' \
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
    assert loss <= 1, f'SSIM loss must be <= 1, got {loss}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test on GPU if there is no GPU.')
def test_ssim_loss_is_less_or_equal_to_one_cuda() -> None:
    ones = torch.ones((3, 3, 256, 256)).cuda()
    zeros = torch.zeros((3, 3, 256, 256)).cuda()
    loss = SSIMLoss()(ones, zeros)
    assert loss <= 1, f'SSIM loss must be <= 1, got {loss}'


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


def test_ssim_loss_raise_if_wrong_value_is_estimated(prediction: torch.Tensor, target: torch.Tensor) -> None:
    ssim_loss = SSIMLoss(kernel_size=11, kernel_sigma=1.5, data_range=1.)(prediction, target)
    tf_prediction = tf.convert_to_tensor(prediction.permute(0, 2, 3, 1).numpy())
    tf_target = tf.convert_to_tensor(target.permute(0, 2, 3, 1).numpy())
    tf_ssim = torch.tensor(tf.image.ssim(tf_prediction, tf_target, max_val=1.).numpy()).mean()
    assert torch.isclose(ssim_loss, tf_ssim, atol=1e-6).all(), \
        f'The estimated value must be equal to tensorflow provided one' \
        f'(considering floating point operation error up to 1 * 10^-6), ' \
        f'got difference {ssim_loss - tf_ssim}'


# ================== Test function: `multi_scale_ssim` ==================
def test_multi_scale_ssim_symmetry(prediction: torch.Tensor, target: torch.Tensor) -> None:
    measure = multi_scale_ssim(prediction, target, data_range=1.)
    reverse_measure = multi_scale_ssim(target, prediction, data_range=1.)
    assert measure == reverse_measure, f'Expect: SSIM(a, b) == SSIM(b, a), got {measure} != {reverse_measure}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test on GPU if there is no GPU.')
def test_multi_scale_ssim_symmetry_cuda(prediction: torch.Tensor, target: torch.Tensor) -> None:
    prediction = prediction.cuda()
    target = target.cuda()
    test_multi_scale_ssim_loss_symmetry(prediction=prediction, target=target)


def test_multi_scale_ssim_measure_is_zero_for_equal_tensors(target: torch.Tensor) -> None:
    prediction = target.clone()
    measure = multi_scale_ssim(prediction, target, data_range=1.)
    measure -= 1.
    assert measure.sum() <= 1e-6, f'If equal tensors are passed SSIM must be equal to 0 ' \
                                  f'(considering floating point operation error up to 1 * 10^-6), got {measure}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test on GPU if there is no GPU.')
def test_multi_scale_ssim_measure_is_zero_for_equal_tensors_cuda(target: torch.Tensor) -> None:
    target = target.cuda()
    test_multi_scale_ssim_measure_is_zero_for_equal_tensors(target=target)


def test_multi_scale_ssim_measure_is_less_or_equal_to_one() -> None:
    # Create two maximally different tensors.
    ones = torch.ones((3, 3, 256, 256))
    zeros = torch.zeros((3, 3, 256, 256))
    measure = multi_scale_ssim(ones, zeros, data_range=1.)
    assert measure <= 1, f'SSIM must be <= 1, got {measure}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test on GPU if there is no GPU.')
def test_multi_scale_ssim_measure_is_less_or_equal_to_one_cuda() -> None:
    ones = torch.ones((3, 3, 256, 256)).cuda()
    zeros = torch.zeros((3, 3, 256, 256)).cuda()
    measure = multi_scale_ssim(ones, zeros, data_range=1.)
    assert measure <= 1, f'SSIM must be <= 1, got {measure}'


def test_multi_scale_ssim_raises_if_tensors_have_different_shapes(target: torch.Tensor) -> None:
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


def test_multi_scale_ssim_raises_if_tensors_have_different_types(target: torch.Tensor) -> None:
    wrong_type_prediction = list(range(10))
    with pytest.raises(AssertionError):
        multi_scale_ssim(wrong_type_prediction, target)


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


def test_multi_scale_ssim_raise_if_wrong_value_is_estimated(prediction: torch.Tensor, target: torch.Tensor) -> None:
    photosynthesis_ms_ssim = multi_scale_ssim(prediction, target, kernel_size=11, kernel_sigma=1.5,
                                              data_range=1., size_average=False)
    tf_prediction = tf.convert_to_tensor(prediction.permute(0, 2, 3, 1).numpy())
    tf_target = tf.convert_to_tensor(target.permute(0, 2, 3, 1).numpy())
    tf_ms_ssim = torch.tensor(tf.image.ssim_multiscale(tf_prediction, tf_target, max_val=1.).numpy())
    assert torch.isclose(photosynthesis_ms_ssim, tf_ms_ssim, atol=1e-4).all(), \
        f'The estimated value must be equal to tensorflow provided one' \
        f'(considering floating point operation error up to 1 * 10^-4), ' \
        f'got difference {photosynthesis_ms_ssim - tf_ms_ssim}'


# ================== Test class: `MultiScaleSSIMLoss` ==================
def test_multi_scale_ssim_loss_symmetry(prediction: torch.Tensor, target: torch.Tensor) -> None:
    loss = MultiScaleSSIMLoss()
    loss_value = loss(prediction, target)
    reverse_loss_value = loss(target, prediction)
    assert loss_value == reverse_loss_value, \
        f'Expect: SSIM(a, b) == SSIM(b, a), got {loss_value} != {reverse_loss_value}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test on GPU if there is no GPU.')
def test_multi_scale_ssim_loss_symmetry_cuda(prediction: torch.Tensor, target: torch.Tensor) -> None:
    prediction = prediction.cuda()
    target = target.cuda()
    test_multi_scale_ssim_loss_symmetry(prediction=prediction, target=target)


def test_multi_scale_ssim_loss_equality(target: torch.Tensor) -> None:
    prediction = target.clone()
    loss = MultiScaleSSIMLoss()(prediction, target)
    loss -= 1.
    assert loss.sum() <= 1e-6, f'If equal tensors are passed SSIM loss must be equal to 0 ' \
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
    assert loss <= 1, f'SSIM loss must be <= 1, got {loss}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test on GPU if there is no GPU.')
def test_multi_scale_ssim_loss_is_less_or_equal_to_one_cuda() -> None:
    ones = torch.ones((3, 3, 256, 256)).cuda()
    zeros = torch.zeros((3, 3, 256, 256)).cuda()
    loss = MultiScaleSSIMLoss()(ones, zeros)
    assert loss <= 1, f'SSIM loss must be <= 1, got {loss}'


def test_multi_scale_ssim_loss_raises_if_tensors_have_different_shapes(target: torch.Tensor) -> None:
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


def test_multi_scale_ssim_loss_raises_if_tensors_have_different_types(target: torch.Tensor) -> None:
    wrong_type_prediction = list(range(10))
    with pytest.raises(AssertionError):
        MultiScaleSSIMLoss()(wrong_type_prediction, target)


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


def test_multi_scale_ssim_loss_raise_if_wrong_value_is_estimated(prediction: torch.Tensor,
                                                                 target: torch.Tensor) -> None:
    photosynthesis_ms_ssim_loss = MultiScaleSSIMLoss(kernel_size=11, kernel_sigma=1.5,
                                                     data_range=1.)(prediction, target)
    tf_prediction = tf.convert_to_tensor(prediction.permute(0, 2, 3, 1).numpy())
    tf_target = tf.convert_to_tensor(target.permute(0, 2, 3, 1).numpy())
    tf_ms_ssim = torch.tensor(tf.image.ssim_multiscale(tf_prediction, tf_target, max_val=1.).numpy()).mean()
    assert torch.isclose(photosynthesis_ms_ssim_loss, tf_ms_ssim, atol=1e-4).all(), \
        f'The estimated value must be equal to tensorflow provided one' \
        f'(considering floating point operation error up to 1 * 10^-4), ' \
        f'got difference {photosynthesis_ms_ssim_loss - tf_ms_ssim}'
