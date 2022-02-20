import torch
import pytest
from typing import Any, Tuple, Callable, Union
from contextlib import contextmanager

from skimage.io import imread
from piq import ContentLoss, StyleLoss, LPIPS, DISTS
from piq.feature_extractors import InceptionV3


@contextmanager
def raise_nothing():
    yield


NONE_GRAD_ERR_MSG = 'Expected non None gradient of leaf variable'


# ================== Test class: `ContentLoss` ==================
def test_content_loss_init() -> None:
    ContentLoss()


def test_content_loss_forward(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    x, y = input_tensors
    loss = ContentLoss()
    loss(x.to(device), y.to(device))


def test_content_loss_computes_grad(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    x, y = input_tensors
    x.requires_grad_()
    loss_value = ContentLoss()(x.to(device), y.to(device))
    loss_value.backward()
    assert x.grad is not None, NONE_GRAD_ERR_MSG


def test_content_loss_raises_if_wrong_reduction(x, y) -> None:
    for mode in ['mean', 'sum', 'none']:
        ContentLoss(reduction=mode)(x, y)

    for mode in [None, 'n', 2]:
        with pytest.raises(ValueError):
            ContentLoss(reduction=mode)(x, y)


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
def test_content_loss_raises_if_wrong_extractor(x, y, model: Union[str, Callable], expectation: Any) -> None:
    with expectation:
        ContentLoss(feature_extractor=model)


@pytest.mark.parametrize(
    "model", ['vgg16', InceptionV3()],
)
def test_content_loss_replace_pooling(x, y, model: Union[str, Callable]) -> None:
    ContentLoss(feature_extractor=model, replace_pooling=True)


def test_content_loss_supports_custom_extractor(x, y, device: str) -> None:
    loss = ContentLoss(feature_extractor=InceptionV3().blocks, layers=['0', '1'], weights=[0.5, 0.5])
    loss(x, y)


@pytest.mark.parametrize(
    "x, y, expectation, value",
    [
        (torch.rand(2, 3, 96, 96, 2), torch.rand(2, 3, 96, 96, 2), pytest.raises(AssertionError), None),
        (torch.randn(2, 3, 96, 96), torch.randn(2, 3, 96, 96), raise_nothing(), None),
        (torch.zeros(2, 3, 96, 96), torch.zeros(2, 3, 96, 96), raise_nothing(), 0.0),
        (torch.ones(2, 3, 96, 96), torch.ones(2, 3, 96, 96), raise_nothing(), 0.0),
        (torch.rand(2, 3, 28, 28), torch.rand(2, 3, 28, 28), pytest.raises(RuntimeError), None),
    ],
)
def test_content_loss_forward_for_special_cases(x, y, expectation: Any, value: float) -> None:
    loss = ContentLoss()
    with expectation:
        if value is None:
            loss(x, y)
        else:
            loss_value = loss(x, y)
            assert torch.isclose(loss_value, torch.tensor(value)), \
                f'Expected loss value to be equal to target value. Got {loss_value} and {value}'


@pytest.mark.skip("Negative tensors are not supported yet")
def test_content_loss_forward_for_normalized_input(device: str) -> None:
    x = torch.randn(2, 3, 96, 96).to(device)
    y = torch.randn(2, 3, 96, 96).to(device)
    loss = ContentLoss(mean=[0., 0., 0.], std=[1., 1., 1.])
    loss(x.to(device), y.to(device))


def test_content_loss_raises_if_layers_weights_mismatch(x, y) -> None:
    wrong_combinations = (
        {
            'layers': ['layer1'],
            'weights': [0.5, 0.5]
        },
        {
            'layers': ['layer1', 'layer2'],
            'weights': [0.5]
        },
        {
            'layers': ['layer1'],
            'weights': []
        }
    )
    for combination in wrong_combinations:
        with pytest.raises(ValueError):
            ContentLoss(**combination)


def test_content_loss_doesnt_rise_if_layers_weights_mismatch_but_allowed(x, y) -> None:
    wrong_combinations = (
        {
            'layers': ['relu1_2'],
            'weights': [0.5, 0.5],
            'allow_layers_weights_mismatch': True
        },
        {
            'layers': ['relu1_2', 'relu2_2'],
            'weights': [0.5],
            'allow_layers_weights_mismatch': True
        },
        {
            'layers': ['relu2_2'],
            'weights': [],
            'allow_layers_weights_mismatch': True
        }
    )
    for combination in wrong_combinations:
        ContentLoss(**combination)


# ================== Test class: `StyleLoss` ==================
def test_style_loss_init() -> None:
    StyleLoss()


def test_style_loss_forward(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    x, y = input_tensors
    loss = StyleLoss()
    loss(x.to(device), y.to(device))


def test_style_loss_computes_grad(input_tensors: Tuple[torch.Tensor, torch.Tensor], device: str) -> None:
    x, y = input_tensors
    x.requires_grad_()
    loss_value = StyleLoss()(x.to(device), y.to(device))
    loss_value.backward()
    assert x.grad is not None, NONE_GRAD_ERR_MSG


def test_style_loss_raises_if_wrong_reduction(x, y) -> None:
    for mode in ['mean', 'sum', 'none']:
        StyleLoss(reduction=mode)(x, y)

    for mode in [None, 'n', 2]:
        with pytest.raises(ValueError):
            StyleLoss(reduction=mode)(x, y)

