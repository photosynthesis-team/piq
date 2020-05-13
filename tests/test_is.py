import torch
import pytest

from photosynthesis_metrics import IS

@pytest.fixture(scope='module')
def features_target_normal() -> torch.Tensor:
    return torch.rand(1000, 20)


@pytest.fixture(scope='module')
def features_prediction_normal() -> torch.Tensor:
    return torch.rand(1000, 20)


@pytest.fixture(scope='module')
def features_prediction_beta() -> torch.Tensor:
    m = torch.distributions.Beta(torch.FloatTensor([2]), torch.FloatTensor([2]))
    return m.sample([1000, 20]).squeeze()


@pytest.fixture(scope='module')
def features_prediction_constant() -> torch.Tensor:
    return torch.ones(1000, 20)

def test_IS_init() -> None:
    try:
        metric = IS()
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_IS_forward(features_target_normal : torch.Tensor, features_prediction_normal : torch.Tensor,) -> None:
    try:
        metric = IS()
        score = metric(features_target_normal, features_prediction_normal)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")

