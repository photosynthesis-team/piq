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

def test_tv_zero_for_equal_tensors(prediction: torch.Tensor):
    l = TVLoss()
    target = prediction.clone()
    measure = l(prediction, target)
    assert measure <= 1e-6, f'TV for equal tensors must be 0, got {measure}'
    

def test_tv_for_known_answer():
    # Tensor with `l1` TV = (10 - 1) * 2  * 2 = 36
    prediction = torch.eye(10).reshape((1, 1, 10, 10))
    # Tensor with TV = 0
    target = torch.zeros((1, 1, 10, 10))
    l = TVLoss(reduction_type='l1')
    measure = l(prediction, target)
    assert measure == 36. , f'TV for this tensors must be 36., got {measure}'
    