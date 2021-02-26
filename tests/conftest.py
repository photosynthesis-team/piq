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
def x() -> torch.Tensor:
    return torch.rand(4, 3, 96, 96)


@pytest.fixture(scope='module')
def y() -> torch.Tensor:
    return torch.rand(4, 3, 96, 96)


x_tensors = [
    torch.rand(4, 3, 96, 96),  # Random 4D
    torch.rand(4, 1, 96, 96),  # Random 4D greyscale
]

y_tensors = [
    torch.rand(4, 3, 96, 96),  # Random 4D
    torch.rand(4, 1, 96, 96),  # Random 4D greyscale
]


@pytest.fixture(params=zip(x_tensors, y_tensors))
def input_tensors(request: Any) -> Any:
    return request.param
