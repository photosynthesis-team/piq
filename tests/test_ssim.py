import torch
import numpy as np
import pytest

from photosynthesis_metrics import SSIMLoss, MultiScaleSSIMLoss


@pytest.fixture(scope='module')
def prediction() -> torch.Tensor:
    return torch.rand(3, 3, 256, 256)


@pytest.fixture(scope='module')
def target() -> torch.Tensor:
    return torch.rand(3, 3, 256, 256)


# ================== Test class: `SSIMLoss` ==================
def test_SSIM_measure_is_zero_for_equal_tensors(target: torch.Tensor) -> None:
    prediction = target.clone()
    measure = SSIMLoss()(prediction, target)
    measure -= 1.
    assert measure.sum() <= 1e-6, f'If equal tensors are passed SSIM must be equal to 0 ' \
                                  f'(considering floating point operation error up to 1 * 10^-6), got {measure}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test on GPU if there is no GPU.')
def test_SSIM_measure_is_zero_for_equal_tensors_cuda(target: torch.Tensor) -> None:
    target = target.cuda()
    test_SSIM_measure_is_zero_for_equal_tensors(target=target)


def test_SSIM_measure_is_less_or_equal_to_one() -> None:
    # Create two maximally different tensors.
    ones = torch.ones((3, 3, 256, 256))
    zeros = torch.zeros((3, 3, 256, 256))
    measure = SSIMLoss()(ones, zeros)
    assert measure <= 1, f'SSIM must be <= 1, got {measure}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test on GPU if there is no GPU.')
def test_SSIM_measure_is_less_or_equal_to_one_cuda() -> None:
    ones = torch.ones((3, 3, 256, 256)).cuda()
    zeros = torch.zeros((3, 3, 256, 256)).cuda()
    measure = SSIMLoss()(ones, zeros)
    assert measure <= 1, f'SSIM must be <= 1, got {measure}'


def test_ssim_raises_if_tensors_have_different_shapes(target: torch.Tensor) -> None:
    wrong_shape_prediction = torch.rand(3, 2, 64, 64)
    with pytest.raises(AssertionError):
        SSIMLoss()(wrong_shape_prediction, target)


def test_ssim_raises_if_tensors_have_different_types(target: torch.Tensor) -> None:
    wrong_type_prediction = np.array(range(10))
    with pytest.raises(AttributeError):
        SSIMLoss()(wrong_type_prediction, target)


def test_ssim_raises_if_wrong_kernel_size_is_passed(prediction: torch.Tensor, target: torch.Tensor) -> None:
    wrong_kernel_sizes = np.array(range(0, 50, 2))
    for kernel_size in wrong_kernel_sizes:
        with pytest.raises(AssertionError):
            SSIMLoss(kernel_size=kernel_size)(prediction, target)


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


# ================== Test class: `MultiScaleSSIMLoss` ==================
def test_multi_scale_ssim_symmetry(prediction: torch.Tensor, target: torch.Tensor) -> None:
    measure = MultiScaleSSIMLoss()(prediction, target)
    reverse_measure = MultiScaleSSIMLoss()(target, prediction)
    assert measure == reverse_measure, f'Expect: SSIM(a, b) == SSIM(b, a), got {measure} != {reverse_measure}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test on GPU if there is no GPU.')
def test_multi_scale_ssim_symmetry_cuda(prediction: torch.Tensor, target: torch.Tensor) -> None:
    prediction = prediction.cuda()
    target = target.cuda()
    test_multi_scale_ssim_loss_symmetry(prediction=prediction, target=target)


def test_multi_scale_ssim_measure_is_zero_for_equal_tensors(target: torch.Tensor) -> None:
    prediction = target.clone()
    measure = MultiScaleSSIMLoss()(prediction, target)
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
    measure = MultiScaleSSIMLoss()(ones, zeros)
    assert measure <= 1, f'SSIM must be <= 1, got {measure}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test on GPU if there is no GPU.')
def test_multi_scale_ssim_measure_is_less_or_equal_to_one_cuda() -> None:
    ones = torch.ones((3, 3, 256, 256)).cuda()
    zeros = torch.zeros((3, 3, 256, 256)).cuda()
    measure = MultiScaleSSIMLoss()(ones, zeros)
    assert measure <= 1, f'SSIM must be <= 1, got {measure}'


def test_multi_scale_ssim_raises_if_tensors_have_different_shapes(target: torch.Tensor) -> None:
    wrong_shape_prediction = torch.rand(3, 2, 64, 64)
    with pytest.raises(AssertionError):
        MultiScaleSSIMLoss()(wrong_shape_prediction, target)


def test_multi_scale_ssim_raises_if_tensors_have_different_types(target: torch.Tensor) -> None:
    wrong_type_prediction = np.array(range(10))
    with pytest.raises(AttributeError):
        MultiScaleSSIMLoss()(wrong_type_prediction, target)


def test_multi_scale_ssim_raises_if_wrong_kernel_size_is_passed(prediction: torch.Tensor, target: torch.Tensor) -> None:
    wrong_kernel_sizes = np.array(range(0, 50, 2))
    for kernel_size in wrong_kernel_sizes:
        with pytest.raises(AssertionError):
            MultiScaleSSIMLoss(kernel_size=kernel_size)(prediction, target)


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
