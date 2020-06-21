import torch
import pytest
from libsvm import svmutil  # noqa: F401
from brisque import BRISQUE
from piq import brisque, BRISQUELoss


@pytest.fixture(scope='module')
def prediction_grey() -> torch.Tensor:
    return torch.rand(3, 1, 256, 256)


@pytest.fixture(scope='module')
def prediction_RGB() -> torch.Tensor:
    return torch.rand(3, 3, 256, 256)


# ================== Test function: `brisque` ==================
def test_brisque_if_works_with_grey(prediction_grey: torch.Tensor) -> None:
    brisque(prediction_grey)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test if there is no GPU.')
def test_brisque_if_works_with_grey_on_gpu(prediction_grey: torch.Tensor) -> None:
    prediction_grey = prediction_grey.cuda()
    brisque(prediction_grey)


def test_brisque_if_works_with_RGB(prediction_RGB: torch.Tensor) -> None:
    brisque(prediction_RGB)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test if there is no GPU.')
def test_brisque_if_works_with_RGB_on_gpu(prediction_RGB: torch.Tensor) -> None:
    prediction_RGB = prediction_RGB.cuda()
    brisque(prediction_RGB)


def test_brisque_raises_if_wrong_reduction(prediction_grey: torch.Tensor) -> None:
    for mode in ['mean', 'sum', 'none']:
        brisque(prediction_grey, reduction=mode)

    for mode in [None, 'n', 2]:
        with pytest.raises(KeyError):
            brisque(prediction_grey, reduction=mode)


def test_brisque_values_grey(prediction_grey: torch.Tensor) -> None:
    score = brisque(prediction_grey, reduction='none')
    score_baseline = torch.tensor([BRISQUE().get_score(img.squeeze().numpy()) for img in prediction_grey])
    assert torch.isclose(score, score_baseline, atol=1e-1).all(), f'Expected values to be equal to ' \
                                                                  f'baseline prediction.' \
                                                                  f'got {score} and {score_baseline}'


def test_brisque_values_RGB(prediction_RGB: torch.Tensor) -> None:
    score = brisque(prediction_RGB, reduction='none')
    score_baseline = torch.tensor([BRISQUE().get_score(img.squeeze().permute(1, 2, 0).numpy()[..., ::-1])
                                   for img in prediction_RGB])
    assert torch.isclose(score, score_baseline, atol=1e-1).all(), f'Expected values to be equal to ' \
                                                                  f'baseline prediction.' \
                                                                  f'got {score} and {score_baseline}'


def test_brisque_all_zeros_or_ones() -> None:
    size = (1, 1, 256, 256)
    for tensor in [torch.zeros(size), torch.ones(size)]:
        with pytest.raises(AssertionError):
            brisque(tensor, reduction='mean')


# ================== Test class: `BRISQUELoss` ==================
def test_brisque_loss_if_works_with_grey(prediction_grey: torch.Tensor) -> None:
    prediction_grey_grad = prediction_grey.clone()
    prediction_grey_grad.requires_grad_()
    loss_value = BRISQUELoss()(prediction_grey_grad)
    loss_value.backward()
    assert prediction_grey_grad.grad is not None, 'Expected non None gradient of leaf variable'


def test_brisque_loss_if_works_with_RGB(prediction_RGB: torch.Tensor) -> None:
    prediction_RGB_grad = prediction_RGB.clone()
    prediction_RGB_grad.requires_grad_()
    loss_value = BRISQUELoss()(prediction_RGB_grad)
    loss_value.backward()
    assert prediction_RGB_grad.grad is not None, 'Expected non None gradient of leaf variable'


def test_brisque_loss_raises_if_wrong_reduction(prediction_grey: torch.Tensor) -> None:
    for mode in ['mean', 'sum', 'none']:
        BRISQUELoss(reduction=mode)(prediction_grey)

    for mode in [None, 'n', 2]:
        with pytest.raises(KeyError):
            BRISQUELoss(reduction=mode)(prediction_grey)
