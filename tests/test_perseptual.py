import torch
import pytest
from typing import Any, List

from photosynthesis_metrics import ContentLoss, StyleLoss, LPIPS


@pytest.fixture(scope='module')
def prediction() -> torch.Tensor:
    return torch.rand(4, 3, 128, 128)


@pytest.fixture(scope='module')
def target() -> torch.Tensor:
    return torch.rand(4, 3, 128, 128)


devices: List[str] = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")


@pytest.fixture(scope='module', params=devices)
def device(request: Any) -> Any:
    return request.param


# ================== Test class: `ContentLoss` ==================
def test_content_loss_init(prediction: torch.Tensor, target: torch.Tensor) -> None:
    ContentLoss()


def test_content_loss_forward(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    loss = ContentLoss()
    loss(prediction.to(device), target.to(device))


# ================== Test class: `StyleLoss` ==================
def test_style_loss_init(prediction: torch.Tensor, target: torch.Tensor) -> None:
    StyleLoss()


def test_style_loss_forward(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    loss = StyleLoss()
    loss(prediction.to(device), target.to(device))


# ================== Test class: `LPIPS` ==================
def test_lpips_loss_init(prediction: torch.Tensor, target: torch.Tensor) -> None:
    LPIPS()


def test_lpips_loss_forward(prediction: torch.Tensor, target: torch.Tensor, device: str) -> None:
    loss = LPIPS()
    loss(prediction.to(device), target.to(device))
