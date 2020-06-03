import torch
import pytest

from photosynthesis_metrics import TVLoss, total_variation


@pytest.fixture(scope='module')
def prediction() -> torch.Tensor:
    return torch.rand(4, 3, 256, 256)


# ================== Test method: `total_variation` ==================
def test_tv_works(prediction: torch.Tensor) -> None:
    for mode in ['l2', 'l1', 'l2_squared']:
        try:
            measure = total_variation(prediction, size_average=True, reduction_type=mode)
        except Exception as e:
            pytest.fail(f"Unexpected error occurred: {e}")
        assert measure > 0


# ================== Test class: `TVLoss` ==================
def test_tv_loss_init() -> None:
    try:
        TVLoss()
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_tv_loss_greater_than_zero(prediction: torch.Tensor) -> None:
    for mode in ['l2', 'l1', 'l2_squared']:
        try:
            res = TVLoss(reduction_type=mode)(prediction)
        except Exception as e:
            pytest.fail(f"Unexpected error occurred: {e}")
        assert res > 0


def test_tv_loss_raises_if_tensors_have_different_types() -> None:
    wrong_type_prediction = list(range(10))
    with pytest.raises(AssertionError):
        TVLoss()(wrong_type_prediction)


def test_tv_loss_check_available_dimensions() -> None:
    custom_prediction = torch.rand(256, 256)
    for _ in range(10):
        if custom_prediction.dim() < 5:
            try:
                TVLoss()(custom_prediction)
            except Exception as e:
                pytest.fail(f"Unexpected error occurred: {e}")
        else:
            with pytest.raises(AssertionError):
                TVLoss()(custom_prediction)
        custom_prediction.unsqueeze_(0)


def test_tv_loss_for_known_answer():
    # Tensor with `l1` TV = (10 - 1) * 2  * 2 = 36
    prediction = torch.eye(10).reshape((1, 1, 10, 10))
    loss = TVLoss(reduction_type='l1')
    measure = loss(prediction)
    assert measure == 36., f'TV for this tensors must be 36., got {measure}'
