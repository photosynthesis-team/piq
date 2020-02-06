import torch
import pytest

from photosynthesis_metrics import MSID


@pytest.fixture(scope='module')
def prediction() -> torch.Tensor:
    return torch.rand(3, 3, 256, 256)


@pytest.fixture(scope='module')
def target() -> torch.Tensor:
    return torch.rand(3, 3, 256, 256)


# ================== Test class: `MSID` ==================
def test_msid_init_with_default_args(prediction: torch.Tensor, target: torch.Tensor) -> None:
    msid = MSID()


def test_msid_is_small_for_equal_tensors(target: torch.Tensor) -> None:
    msid = MSID()
    prediction = target.clone()
    measure = msid(prediction, target)
    assert measure.sum() <= 10, f'If equal tensors are passed MSID must be small, , got {measure}'
