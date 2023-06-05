import torch
import pytest
import os
import hashlib
import re

import numpy as np

from piq.utils.common import _validate_input, _reduce, _parse_version, download_tensor, is_sha256_hash


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
        _validate_input([tensor_2d, tensor_5d], dim_range=(2, 5), check_for_channels_first=True)


def test_breaks_if_wrong_channel_order() -> None:
    with pytest.raises(AssertionError):
        _validate_input([torch.rand(1, 5, 5, 1)], check_for_channels_first=True)
        _validate_input([torch.rand(1, 5, 5, 2)], check_for_channels_first=True)
        _validate_input([torch.rand(1, 5, 5, 3)], check_for_channels_first=True)


def test_works_if_correct_channel_order() -> None:
    try:
        _validate_input([torch.rand(1, 1, 5, 5)], check_for_channels_first=True)
        _validate_input([torch.rand(1, 2, 5, 5)], check_for_channels_first=True)
        _validate_input([torch.rand(1, 3, 5, 5)], check_for_channels_first=True)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


# ================== Test function: `_reduce` ==================
def test_reduce_function() -> None:
    x = torch.rand(1, 1, 1, 1)
    for reduction in ['mean', 'sum', 'none']:
        _reduce(x, reduction=reduction)

    for reduction in [None, 'n', 2]:
        with pytest.raises(ValueError):
            _reduce(x, reduction=reduction)


# =============== Test function: `_parse_version` ==============
# Test cases are examples of valid semver versioning options
# See https://semver.org/ for more details.
# Test cases are inspired by https://github.com/python-semver
valid_semver_versions = ("version,expected",
                         [
                             # no. 1
                             ("1.2.3-alpha.1.2+build.11.e0f985a", (1, 2, 3)),
                             # no. 2
                             ("1.2.3-alpha-1+build.11.e0f985a", (1, 2, 3)),
                             # no. 3
                             ("0.1.0-0f", (0, 1, 0)),
                             # no. 4
                             ("0.0.0-0foo.1", (0, 0, 0)),
                             # no. 5
                             ("0.0.0-0foo.1+build.1", (0, 0, 0)),
                             # no. 6
                             ("1.3.5-alpha", (1, 3, 5)),
                             # no. 7
                             ("1.3.5", (1, 3, 5)),
                             # no. 8
                             ("1.10.0a0+ecc3718", (1, 10, 0)),
                             # no. 9
                             ("1.8.0a0+17f8c32", (1, 8, 0))
                         ])


@pytest.mark.parametrize(*valid_semver_versions)
def test_version_tuple_doesnt_fail_valid_input(version, expected) -> None:
    try:
        _parse_version(version)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred while parsing valid semver versions: {e}")


@pytest.mark.parametrize(*valid_semver_versions)
def test_version_tuple_parses_correctly(version, expected) -> None:
    parsed = _parse_version(version)
    assert parsed == expected, "Wrong parsing result of a valid semver version"


@pytest.mark.parametrize("version", ["01a.2.3", "1.2.03a.a.a.a", "a.1.3.5.post1"])
def test_version_tuple_warns_on_invalid_input(version) -> None:
    with pytest.warns(UserWarning):
        _parse_version(version)


def test_download_tensor():
    url = "https://github.com/photosynthesis-team/piq/releases/download/v0.7.1/clipiqa_tokens.pt"
    file_name = os.path.basename(url)
    root = os.path.expanduser("~/.cache/clip")

    # Check if tensor gets downloaded if not cached locally.
    full_file_path = os.path.join(root, file_name)
    print('full_file_path', full_file_path)
    if os.path.exists(full_file_path):
        os.remove(full_file_path)

    assert isinstance(download_tensor(url, root), torch.Tensor)

    # Check if tensor loads if cached.
    assert isinstance(download_tensor(url, root), torch.Tensor)


# =============== Test function: `is_sha256_hash` ==============
def test_works_for_hashes():
    example_stings = [b'the', b'the', b'meaning', b'of', b'life', b'the', b'universe', b'and', b'everything']
    for ex in example_stings:
        h = hashlib.new('sha256')
        h.update(ex)
        h = h.hexdigest()
        assert isinstance(is_sha256_hash(h), re.Match), f'Exepected re.Match, got {type(h)}'


def test_does_not_work_for_plane_strings():
    example_stings = ['the', 'the', 'meaning', 'of', 'life', 'the' 'universe', 'and' 'everything']
    for ex in example_stings:
        with pytest.raises(AssertionError):
            assert isinstance(ex, re.Match), f'Exepected re.Match, got {type(hash)}'
