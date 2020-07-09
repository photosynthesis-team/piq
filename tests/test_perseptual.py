import torch
import pytest
from typing import Any, Tuple, Callable, Union
from contextlib import contextmanager

from skimage.io import imread
from piq import ContentLoss, StyleLoss, LPIPS, DISTS
from piq.feature_extractors.fid_inception import InceptionV3


@contextmanager
def raise_nothing():
    yield


# ================== Test class: `ContentLoss` ==================
def test_content_loss_init() -> None:
    ContentLoss()


def test_content_loss_forward(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction, target = input_tensors
    loss = ContentLoss()
    loss(prediction.to(device), target.to(device))


def test_content_loss_backward(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction, target = input_tensors
    prediction.requires_grad_()
    loss = ContentLoss()
    loss(prediction.to(device), target.to(device)).backward()


def test_content_loss_raises_if_wrong_reduction(prediction: torch.Tensor, target: torch.Tensor) -> None:
    for mode in ['mean', 'sum', 'none']:
        ContentLoss(reduction=mode)(prediction, target)

    for mode in [None, 'n', 2]:
        with pytest.raises(KeyError):
            ContentLoss(reduction=mode)(prediction, target)


@pytest.mark.parametrize(
    "model,expectation",
    [
        ('vgg16', raise_nothing()),
        ('vgg19', raise_nothing()),
        (InceptionV3(), raise_nothing()),
        (None, pytest.raises(ValueError)),
        ('random_encoder', pytest.raises(ValueError)),
    ],
)
def test_content_loss_raises_if_wrong_extractor(
        prediction: torch.Tensor, target: torch.Tensor, model: Union[str, Callable], expectation: Any) -> None:
    with expectation:
        ContentLoss(feature_extractor=model)


@pytest.mark.parametrize(
    "model", ['vgg16', InceptionV3()],
)
def test_content_loss_replace_pooling(
        prediction: torch.Tensor, target: torch.Tensor, model: Union[str, Callable]) -> None:
    ContentLoss(feature_extractor=model, replace_pooling=True)


def test_content_loss_supports_custom_extractor(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    loss = ContentLoss(feature_extractor=InceptionV3().blocks, layers=['0', '1'])
    loss(prediction, target)


@pytest.mark.parametrize(
    "prediction,target,expectation,value",
    [
        (torch.rand(4, 3, 96, 96, 2), torch.rand(4, 3, 96, 96, 2), pytest.raises(AssertionError), None),
        (torch.randn(4, 3, 96, 96), torch.randn(4, 3, 96, 96), pytest.raises(AssertionError), None),
        (torch.zeros(4, 3, 96, 96), torch.zeros(4, 3, 96, 96), raise_nothing(), 0.0),
        (torch.ones(4, 3, 96, 96), torch.ones(4, 3, 96, 96), raise_nothing(), 0.0),
        (torch.rand(4, 3, 28, 28), torch.rand(4, 3, 28, 28), pytest.raises(RuntimeError), None),
    ],
)
def test_content_loss_forward_for_special_cases(
        prediction: torch.Tensor, target: torch.Tensor, expectation: Any, value: float) -> None:
    loss = ContentLoss()
    with expectation:
        if value is None:
            loss(prediction, target)
        else:
            loss_value = loss(prediction, target)
            assert torch.isclose(loss_value, torch.tensor(value)), \
                f'Expected loss value to be equal to target value. Got {loss_value} and {value}'


# ================== Test class: `StyleLoss` ==================
def test_style_loss_init() -> None:
    StyleLoss()


def test_style_loss_forward(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction, target = input_tensors
    loss = StyleLoss()
    loss(prediction.to(device), target.to(device))


def test_style_loss_backward(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction, target = input_tensors
    prediction.requires_grad_()
    loss = StyleLoss()
    loss(prediction.to(device), target.to(device)).backward()


def test_style_loss_raises_if_wrong_reduction(prediction: torch.Tensor, target: torch.Tensor) -> None:
    for mode in ['mean', 'sum', 'none']:
        StyleLoss(reduction=mode)(prediction, target)

    for mode in [None, 'n', 2]:
        with pytest.raises(KeyError):
            StyleLoss(reduction=mode)(prediction, target)


# ================== Test class: `LPIPS` ==================
def test_lpips_loss_init() -> None:
    LPIPS()


def test_lpips_loss_forward(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction, target = input_tensors
    loss = LPIPS()
    loss(prediction.to(device), target.to(device))


def test_lpips_loss_backward(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    loss = LPIPS()
    loss(prediction.to(device), target.to(device))


def test_lpips_loss_raises_if_wrong_reduction(prediction: torch.Tensor, target: torch.Tensor) -> None:
    for mode in ['mean', 'sum', 'none']:
        LPIPS(reduction=mode)(prediction, target)

    for mode in [None, 'n', 2]:
        with pytest.raises(KeyError):
            LPIPS(reduction=mode)(prediction, target)


# ================== Test class: `DISTS` ==================
def test_dists_loss_forward(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    loss = DISTS()
    loss(prediction.to(device), target.to(device))


def test_dists_loss_backward(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    loss = LPIPS()
    loss(prediction.to(device), target.to(device))


@pytest.mark.parametrize(
    "prediction,target,expectation,value",
    [
        (torch.zeros(4, 3, 128, 128), torch.zeros(4, 3, 128, 128), raise_nothing(), 0.0),
        (torch.ones(4, 3, 128, 128), torch.ones(4, 3, 128, 128), raise_nothing(), 0.0),
    ],
)
def test_dists_loss_forward_for_special_cases(
        prediction: torch.Tensor, target: torch.Tensor, expectation: Any, value: float) -> None:
    loss = ContentLoss()
    with expectation:
        if value is None:
            loss(prediction, target)
        else:
            loss_value = loss(prediction, target)
            assert torch.isclose(loss_value, torch.tensor(value)), \
                f'Expected loss value to be equal to target value. Got {loss_value} and {value}'


def test_dists_simmilar_to_official_implementation() -> None:
    # Baseline scores from: https://github.com/dingkeyan93/DISTS
    loss = DISTS()

    # Greyscale images
    goldhill = torch.tensor(imread('tests/assets/goldhill.gif')) / 255.0
    goldhill_jpeg = torch.tensor(imread('tests/assets/goldhill_jpeg.gif')) / 255.0

    loss_value = loss(goldhill_jpeg, goldhill)
    baseline_value = torch.tensor(0.3447)
    assert torch.isclose(loss_value, baseline_value, atol=1e-3), \
        f'Expected PIQ loss to be equal to original. Got {loss_value} and {baseline_value}'

    # RGB images
    I01 = torch.tensor(imread('tests/assets/I01.BMP')).permute(2, 0, 1) / 255.0
    i1_01_5 = torch.tensor(imread('tests/assets/i01_01_5.bmp')).permute(2, 0, 1) / 255.0

    loss_value = loss(i1_01_5, I01)
    baseline_value = torch.tensor(0.2376)

    assert torch.isclose(loss_value, baseline_value, atol=1e-3), \
        f'Expected PIQ loss to be equal to original. Got {loss_value} and {baseline_value}'
