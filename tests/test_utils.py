import torch
import pytest

import numpy as np

from piq.utils import _validate_input, _reduce


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
            _validate_input([inp, ])


def test_works_on_single_not_5d_tensor(tensor_1d: torch.Tensor) -> None:
    tensor = tensor_1d.clone()
    # 1D -> max_num_dims
    max_num_dims = 10
    for _ in range(max_num_dims):
        if 1 < tensor.dim() < 5:
            try:
                _validate_input([tensor, ], dim_range=(2, 4))
            except Exception as e:
                pytest.fail(f"Unexpected error occurred: {e}")
        else:
            with pytest.raises(AssertionError):
                _validate_input([tensor, ], dim_range=(2, 4))

        tensor.unsqueeze_(0)


def test_works_on_single_5d_tensor(tensor_5d: torch.Tensor) -> None:
    try:
        _validate_input([tensor_5d, ], dim_range=(5, 5))
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_works_on_two_not_5d_tensors(tensor_1d: torch.Tensor) -> None:
    tensor = tensor_1d.clone()
    max_num_dims = 10
    for _ in range(max_num_dims):
        another_tensor = tensor.clone()
        if 1 < tensor.dim() < 5:
            try:
                _validate_input([tensor, another_tensor], dim_range=(2, 4))
            except Exception as e:
                pytest.fail(f"Unexpected error occurred: {e}")
        else:
            with pytest.raises(AssertionError):
                _validate_input([tensor, another_tensor], dim_range=(2, 4))

        tensor.unsqueeze_(0)


def test_breaks_if_tensors_have_different_n_dims(tensor_2d: torch.Tensor, tensor_5d: torch.Tensor) -> None:
    with pytest.raises(AssertionError):
        _validate_input([tensor_2d, tensor_5d], dim_range=(2, 5))


# ================== Test function: `_reduce` ==================
def test_reduce_function() -> None:
    x = torch.rand(1, 1, 1, 1)
    for reduction in ['mean', 'sum', 'none']:
        _reduce(x, reduction=reduction)

    for reduction in [None, 'n', 2]:
        with pytest.raises(ValueError):
            _reduce(x, reduction=reduction)
