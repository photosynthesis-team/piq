import torch
import pytest

from photosynthesis_metrics import total_variation, TVLoss


@pytest.fixture(scope='module')
def prediction() -> torch.Tensor:
    return torch.rand(4, 3, 256, 256)


@pytest.fixture(scope='module')
def target() -> torch.Tensor:
    return torch.rand(4, 3, 256, 256)


# ================== Test function: `total_variation` ==================
def test_tv_works(prediction: torch.Tensor) -> None:
    measure = total_variation(prediction, size_average=True, reduction_type='l2')
    assert measure > 0 
    measure = total_variation(prediction, size_average=True, reduction_type='l1')
    measure = total_variation(prediction, size_average=True, reduction_type='l2_squared')



# ================== Test class: `TVLoss` ==================
def test_tvloss_init(prediction: torch.Tensor, target: torch.Tensor) -> None:
    l = TVLoss()

def test_tvloss(prediction: torch.Tensor, target: torch.Tensor) -> None:
    l = TVLoss()
    res = l(prediction, target)
    assert res > 0
