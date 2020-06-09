import torch
import pytest

import numpy as np

from photosynthesis_metrics.utils import _validate_input


@pytest.fixture(scope='module')
def tensor_1d() -> torch.Tensor:
    return torch.rand(1)


@pytest.fixture(scope='module')
def tensor_2d() -> torch.Tensor:
    return torch.rand((2, 2))


@pytest.fixture(scope='module')
def tensor_5d() -> torch.Tensor:
    return torch.rand((5, 5, 5, 5, 2))


@pytest.fixture(scope='module')
def tensor_5d_broken() -> torch.Tensor:
    return torch.rand((5, 5, 5, 5, 5))


# ================== Test function: `_validate_input` ==================
def test_breaks_if_not_supported_data_types_provided() -> None:
    inputs_of_wrong_types = [[], (), {}, 42, '42', True, 42., np.array([42]), None]
    for inp in inputs_of_wrong_types:
        with pytest.raises(AssertionError):
            _validate_input(inp, allow_5d=False)


def test_breaks_if_too_many_tensors_provided(tensor_2d: torch.Tensor) -> None:
    max_number_of_tensors = 2
    for n_tensors in range(max_number_of_tensors + 1, (max_number_of_tensors + 1) * 10):
        inp = tuple(tensor_2d.clone() for _ in range(n_tensors))
        with pytest.raises(AssertionError):
            _validate_input(inp, allow_5d=False)


def test_works_on_single_not_5d_tensor(tensor_1d: torch.Tensor) -> None:
    tensor = tensor_1d.clone()
    # 1D -> max_num_dims
    max_num_dims = 10
    for _ in range(max_num_dims):
        if 1 < tensor.dim() < 5:
            try:
                _validate_input(tensor, allow_5d=False)
            except Exception as e:
                pytest.fail(f"Unexpected error occurred: {e}")
        else:
            with pytest.raises(AssertionError):
                _validate_input(tensor, allow_5d=False)

        tensor.unsqueeze_(0)


def test_works_on_single_5d_tensor(tensor_5d: torch.Tensor) -> None:
    try:
        _validate_input(tensor_5d, allow_5d=True)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_breaks_if_5d_tensor_has_wrong_format(tensor_5d_broken: torch.Tensor) -> None:
    with pytest.raises(Exception):
        _validate_input(tensor_5d_broken, allow_5d=True)


def test_works_on_two_not_5d_tensors(tensor_1d: torch.Tensor) -> None:
    tensor = tensor_1d.clone()
    max_num_dims = 10
    for _ in range(max_num_dims):
        another_tensor = tensor.clone()
        if 1 < tensor.dim() < 5:
            try:
                _validate_input((tensor, another_tensor), allow_5d=True)
            except Exception as e:
                pytest.fail(f"Unexpected error occurred: {e}")
        else:
            with pytest.raises(AssertionError):
                _validate_input(tensor, allow_5d=False)

        tensor.unsqueeze_(0)


def test_works_on_two_5d_tensors(tensor_5d: torch.Tensor) -> None:
    another_tensor_5d = tensor_5d.clone()
    try:
        _validate_input((tensor_5d, another_tensor_5d), allow_5d=True)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_breaks_if_tensors_have_different_n_dims(tensor_2d: torch.Tensor, tensor_5d: torch.Tensor) -> None:
    with pytest.raises(AssertionError):
        _validate_input((tensor_2d, tensor_5d), allow_5d=True)


def test_works_if_kernel_size_is_odd(tensor_2d: torch.Tensor) -> None:
    for kernel_size in [i * 2 + 1 for i in range(2, 42)]:
        try:
            _validate_input(tensor_2d, allow_5d=False, kernel_size=kernel_size)
        except Exception as e:
            pytest.fail(f"Unexpected error occurred: {e}")


def test_breaks_if_kernel_size_is_even(tensor_2d: torch.Tensor) -> None:
    for kernel_size in [i * 2 for i in range(2, 42)]:
        with pytest.raises(AssertionError):
            _validate_input(tensor_2d, allow_5d=False, kernel_size=kernel_size)


def test_breaks_if_scale_weights_of_not_supported_data_types_provided(tensor_2d: torch.Tensor) -> None:
    wrong_scale_weights = [
        ['1', '2', '3'],
        np.array([1, 2, 3])
    ]
    for weights in wrong_scale_weights:
        with pytest.raises(Exception):
            _validate_input(tensor_2d, allow_5d=False, scale_weights=weights)


def test_breaks_if_scale_weight_wrong_n_dims_provided(tensor_2d: torch.Tensor) -> None:
    wrong_scale_weights = tensor_2d.clone()
    with pytest.raises(AssertionError):
        _validate_input(tensor_2d, allow_5d=False, scale_weights=wrong_scale_weights)
