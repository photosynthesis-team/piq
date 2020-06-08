import torch
import pytest
from libsvm import svmutil  # noqa: F401
from brisque import BRISQUE
from photosynthesis_metrics import brisque, BRISQUELoss


@pytest.fixture(scope='module')
def prediction_grey() -> torch.Tensor:
    return torch.rand(3, 1, 256, 256)


@pytest.fixture(scope='module')
def prediction_RGB() -> torch.Tensor:
    return torch.rand(3, 3, 256, 256)


# ================== Test function: `brisque` ==================
def test_brisque_raises_if_wrong_type_input() -> None:
    with pytest.raises(AssertionError):
        brisque(None)


def test_brisque_if_works_with_1d(prediction_grey: torch.Tensor) -> None:
    try:
        brisque(prediction_grey)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_brisque_if_works_with_3d(prediction_RGB: torch.Tensor) -> None:
    try:
        brisque(prediction_RGB)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_brisque_check_available_dimensions() -> None:
    custom_prediction = torch.rand(256, 256)
    for _ in range(10):
        if custom_prediction.dim() < 5:
            try:
                brisque(custom_prediction)
            except Exception as e:
                pytest.fail(f"Unexpected error occurred: {e}")
        else:
            with pytest.raises(AssertionError):
                brisque(custom_prediction)
        custom_prediction.unsqueeze_(0)


def test_brisque_raises_if_wrong_reduction(prediction_grey: torch.Tensor) -> None:
    for mode in ['mean', 'sum', 'none']:
        try:
            brisque(prediction_grey, reduction=mode)
        except Exception as e:
            pytest.fail(f"Unexpected error occurred: {e}")
    for mode in [None, 'n', 2]:
        with pytest.raises(ValueError):
            brisque(prediction_grey, reduction=mode)


def test_brisque_values_1d(prediction_grey: torch.Tensor) -> None:
    score = brisque(prediction_grey, reduction='none')
    score_baseline = torch.tensor([BRISQUE().get_score(img.squeeze().numpy()) for img in prediction_grey])
    assert torch.isclose(score, score_baseline, rtol=2e-4, atol=1e-6).all(), f'Expected values to be equal to ' \
                                                                             f'baseline prediction.' \
                                                                             f'got {score} and {score_baseline}'


def test_brisque_values_3d(prediction_RGB: torch.Tensor) -> None:
    score = brisque(prediction_RGB, reduction='none')
    score_baseline = torch.tensor([BRISQUE().get_score(img.squeeze().permute(1, 2, 0).numpy()[..., ::-1])
                                   for img in prediction_RGB])
    assert torch.isclose(score, score_baseline, rtol=2e-4).all(), f'Expected values to be equal to ' \
                                                                  f'baseline prediction.' \
                                                                  f'got {score} and {score_baseline}'


# ================== Test class: `BRISQUELoss` ==================
def test_brisque_loss_if_works_with_1d(prediction_grey: torch.Tensor) -> None:
    try:
        BRISQUELoss()(prediction_grey)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_brisque_loss_if_works_with_3d(prediction_RGB: torch.Tensor) -> None:
    try:
        BRISQUELoss()(prediction_RGB)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_brisque_loss_raises_if_wrong_reduction(prediction_grey: torch.Tensor) -> None:
    for mode in ['mean', 'sum', 'none']:
        try:
            BRISQUELoss(reduction=mode)(prediction_grey)
        except Exception as e:
            pytest.fail(f"Unexpected error occurred: {e}")
    for mode in [None, 'n', 2]:
        with pytest.raises(ValueError):
            BRISQUELoss(reduction=mode)(prediction_grey)
