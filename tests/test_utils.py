import torch
import pytest

import numpy as np

from photosynthesis_metrics.utils import _validate_input


@pytest.fixture(scope='module')
def tensor_4d() -> torch.Tensor:
    return torch.rand((4, 4, 4, 4))


@pytest.fixture(scope='module')
def tensor_5d() -> torch.Tensor:
    return torch.rand((5, 5, 5, 5, 2))


@pytest.fixture(scope='module')
def tensor_5d_broken() -> torch.Tensor:
    return torch.rand((5, 5, 5, 5, 5))


# ================== Test function: `_validate_input` ==================
def test_breaks_if_not_supported_data_types_provided() -> None:
    inputs_of_wrong_types = [[], (), {}, 42, '42', True, 42., np.array([42])]
    for inp in inputs_of_wrong_types:
        with pytest.raises(AssertionError):
            _validate_input(inp, supports_5d=False)


def test_breaks_if_too_many_tensors_provided(tensor_4d: torch.Tensor) -> None:
    max_number_of_tensors = 2
    for n_tensors in range(max_number_of_tensors + 1, (max_number_of_tensors + 1) * 10):
        inp = tuple(tensor_4d.clone() for _ in range(n_tensors))
        with pytest.raises(AssertionError):
            _validate_input(inp, supports_5d=False)


def test_works_on_single_4d_tensor(tensor_4d: torch.Tensor) -> None:
    try:
        _validate_input(tensor_4d, supports_5d=False)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_works_on_single_5d_tensor(tensor_5d: torch.Tensor) -> None:
    try:
        _validate_input(tensor_5d, supports_5d=True)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_breaks_if_5d_tensor_has_wrong_format(tensor_5d_broken: torch.Tensor) -> None:
    with pytest.raises(Exception):
        _validate_input(tensor_5d_broken, supports_5d=True)


def test_works_on_two_4d_tensors(tensor_4d: torch.Tensor) -> None:
    another_tensor_4d = tensor_4d.clone()
    try:
        _validate_input((tensor_4d, another_tensor_4d), supports_5d=True)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_works_on_two_5d_tensors(tensor_5d: torch.Tensor) -> None:
    another_tensor_5d = tensor_5d.clone()
    try:
        _validate_input((tensor_5d, another_tensor_5d), supports_5d=True)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_breaks_if_tensors_have_different_n_dims(tensor_4d: torch.Tensor, tensor_5d: torch.Tensor) -> None:
    with pytest.raises(AssertionError):
        _validate_input((tensor_4d, tensor_5d), supports_5d=True)


def test_works_if_kernel_size_is_odd(tensor_4d: torch.Tensor) -> None:
    for kernel_size in [i * 2 + 1 for i in range(2, 42)]:
        _validate_input(tensor_4d, supports_5d=False, kernel_size=kernel_size)


def test_breaks_if_kernel_size_is_even(tensor_4d: torch.Tensor) -> None:
    for kernel_size in [i * 2 for i in range(2, 42)]:
        with pytest.raises(AssertionError):
            _validate_input(tensor_4d, supports_5d=False, kernel_size=kernel_size)


def test_breaks_if_scale_weights_of_not_supported_data_types_provided(tensor_4d: torch.Tensor) -> None:
    wrong_scale_weights = [
        ['1', '2', '3'],
        np.array([1, 2, 3])
    ]
    for weights in wrong_scale_weights:
        with pytest.raises(Exception):
            _validate_input(tensor_4d, supports_5d=False, scale_weights=weights)


def test_breaks_if_scale_weight_wrong_n_dims_provided(tensor_4d: torch.Tensor) -> None:
    wrong_scale_weights = tensor_4d.clone()
    with pytest.raises(AssertionError):
        _validate_input(tensor_4d, supports_5d=False, scale_weights=wrong_scale_weights)
