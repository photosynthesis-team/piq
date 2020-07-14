from typing import List, Any

import pytest
import torch

devices: List[str] = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")


@pytest.fixture(params=devices, scope='module')
def device(request: Any) -> Any:
    return request.param


@pytest.fixture(scope='module')
def prediction() -> torch.Tensor:
    return torch.rand(4, 3, 128, 128)


@pytest.fixture(scope='module')
def target() -> torch.Tensor:
    return torch.rand(4, 3, 128, 128)


prediction_tensors = [
    torch.rand(4, 3, 96, 96),  # Random 4D
    torch.rand(3, 96, 96),  # Random 3D
    torch.rand(4, 1, 96, 96),  # Random 4D greyscale
    torch.rand(96, 96),  # Random 2D greyscale
]

target_tensors = [
    torch.rand(4, 3, 96, 96),  # Random 4D
    torch.rand(3, 96, 96),  # Random 3D
    torch.rand(4, 1, 96, 96),  # Random 4D greyscale
    torch.rand(96, 96),  # Random 2D greyscale
]


@pytest.fixture(params=zip(prediction_tensors, target_tensors))
def input_tensors(request: Any) -> Any:
    return request.param
