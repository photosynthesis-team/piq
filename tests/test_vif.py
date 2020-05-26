import torch
import pytest

from photosynthesis_metrics import VIFLoss, vif_p


@pytest.fixture(scope='module')
def prediction() -> torch.Tensor:
    return torch.rand(4, 3, 256, 256)


@pytest.fixture(scope='module')
def target() -> torch.Tensor:
    return torch.rand(4, 3, 256, 256)


@pytest.fixture(scope='module')
def prediction_1d() -> torch.Tensor:
    return torch.rand(4, 1, 256, 256)


@pytest.fixture(scope='module')
def target_1d() -> torch.Tensor:
    return torch.rand(4, 1, 256, 256)


# ================== Test function: `vif_p` ==================
def test_vif_p_works_for_3_channels(prediction: torch.Tensor, target: torch.Tensor) -> None:
    try:
        vif_p(prediction, target, data_range=1.)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_vif_p_works_for_1_channel(prediction_1d: torch.Tensor, target_1d: torch.Tensor) -> None:
    try:
        vif_p(prediction_1d, target_1d, data_range=1.)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_vif_p_works_for_different_data_range(prediction: torch.Tensor, target: torch.Tensor) -> None:
    prediction_255 = (prediction * 255).type(torch.uint8)
    target_255 = (target * 255).type(torch.uint8)
    try:
        vif_p(prediction_255, target_255, data_range=255)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


# ================== Test class: `VIFLoss` ==================
def test_vif_loss_init(prediction: torch.Tensor, target: torch.Tensor) -> None:
    try:
        VIFLoss()
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_vif_loss_zero_for_equal_tensors(prediction: torch.Tensor):
    loss = VIFLoss()
    target = prediction.clone()
    measure = loss(prediction, target)
    assert measure <= 1e-6, f'VIF for equal tensors must be 0, got {measure}'
