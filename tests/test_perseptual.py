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


NONE_GRAD_ERR_MSG = 'Expected non None gradient of leaf variable'


# ================== Test class: `ContentLoss` ==================
def test_content_loss_init() -> None:
    ContentLoss()


def test_content_loss_forward(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction, target = input_tensors
    loss = ContentLoss()
    loss(prediction.to(device), target.to(device))


def test_content_loss_computes_grad(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction, target = input_tensors
    prediction.requires_grad_()
    loss_value = ContentLoss()(prediction.to(device), target.to(device))
    loss_value.backward()
    assert prediction.grad is not None, NONE_GRAD_ERR_MSG


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
        (torch.rand(2, 3, 96, 96, 2), torch.rand(2, 3, 96, 96, 2), pytest.raises(AssertionError), None),
        (torch.randn(2, 3, 96, 96), torch.randn(2, 3, 96, 96), pytest.raises(AssertionError), None),
        (torch.zeros(2, 3, 96, 96), torch.zeros(2, 3, 96, 96), raise_nothing(), 0.0),
        (torch.ones(2, 3, 96, 96), torch.ones(2, 3, 96, 96), raise_nothing(), 0.0),
        (torch.rand(2, 3, 28, 28), torch.rand(2, 3, 28, 28), pytest.raises(RuntimeError), None),
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


@pytest.mark.skip("Negative tensors are not supported yet")
def test_content_loss_forward_for_normalized_input(device: str) -> None:
    prediction = torch.randn(2, 3, 96, 96).to(device)
    target = torch.randn(2, 3, 96, 96).to(device)
    loss = ContentLoss(mean=[0., 0., 0.], std=[1., 1., 1.])
    loss(prediction.to(device), target.to(device))


# ================== Test class: `StyleLoss` ==================
def test_style_loss_init() -> None:
    StyleLoss()


def test_style_loss_forward(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction, target = input_tensors
    loss = StyleLoss()
    loss(prediction.to(device), target.to(device))


def test_style_loss_computes_grad(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    prediction, target = input_tensors
    prediction.requires_grad_()
    loss_value = StyleLoss()(prediction.to(device), target.to(device))
    loss_value.backward()
    assert prediction.grad is not None, NONE_GRAD_ERR_MSG


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


def test_lpips_computes_grad(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    prediction.requires_grad_()
    loss_value = LPIPS()(prediction.to(device), target.to(device))
    loss_value.backward()
    assert prediction.grad is not None, NONE_GRAD_ERR_MSG


def test_lpips_loss_raises_if_wrong_reduction(prediction: torch.Tensor, target: torch.Tensor) -> None:
    for mode in ['mean', 'sum', 'none']:
        LPIPS(reduction=mode)(prediction, target)

    for mode in [None, 'n', 2]:
        with pytest.raises(KeyError):
            LPIPS(reduction=mode)(prediction, target)


@pytest.mark.parametrize(
    "prediction,target,expectation,value",
    [
        (torch.zeros(2, 3, 96, 96), torch.zeros(2, 3, 96, 96), raise_nothing(), 0.0),
        (torch.ones(2, 3, 96, 96), torch.ones(2, 3, 96, 96), raise_nothing(), 0.0),
    ],
)
def test_lpips_loss_forward_for_special_cases(
        prediction: torch.Tensor, target: torch.Tensor, expectation: Any, value: float) -> None:
    loss = LPIPS()
    with expectation:
        if value is None:
            loss(prediction, target)
        else:
            loss_value = loss(prediction, target)
            assert torch.isclose(loss_value, torch.tensor(value), atol=1e-6), \
                f'Expected loss value to be equal to target value. Got {loss_value} and {value}'


# ================== Test class: `DISTS` ==================
def test_dists_loss_forward(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    loss = DISTS()
    loss(prediction.to(device), target.to(device))


def test_dists_computes_grad(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    prediction.requires_grad_()
    loss_value = DISTS()(prediction.to(device), target.to(device))
    loss_value.backward()
    assert prediction.grad is not None, NONE_GRAD_ERR_MSG


@pytest.mark.parametrize(
    "prediction,target,expectation,value",
    [
        (torch.zeros(2, 3, 96, 96), torch.zeros(2, 3, 96, 96), raise_nothing(), 0.0),
        (torch.ones(2, 3, 96, 96), torch.ones(2, 3, 96, 96), raise_nothing(), 0.0),
    ],
)
def test_dists_loss_forward_for_special_cases(
        prediction: torch.Tensor, target: torch.Tensor, expectation: Any, value: float) -> None:
    loss = DISTS()
    with expectation:
        if value is None:
            loss(prediction, target)
        else:
            loss_value = loss(prediction, target)
            assert torch.isclose(loss_value, torch.tensor(value), atol=1e-6), \
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
