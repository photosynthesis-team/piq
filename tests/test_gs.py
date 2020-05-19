import sys

import pytest
import torch

from photosynthesis_metrics import GS
try:
    import gudhi # noqa
except ImportError:
    pass


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


# ================== Test class: `GS` ==================
@pytest.mark.skipif('gudhi' not in sys.modules, reason="Requires gudhi library")
def test_GS_init() -> None:
    try:
        GS()
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


@pytest.mark.skipif('gudhi' not in sys.modules, reason="Requires gudhi library")
def test_GS_forward(
        features_target_normal: torch.Tensor, features_prediction_normal: torch.Tensor,) -> None:
    try:
        metric = GS()
        metric(features_target_normal, features_prediction_normal)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


@pytest.mark.skipif('gudhi' not in sys.modules, reason="Requires gudhi library")
def test_GS_similar_for_same_distribution(
        features_target_normal: torch.Tensor, features_prediction_normal: torch.Tensor) -> None:
    metric = GS(sample_size=64, num_iters=500, i_max=100, num_workers=4)
    diff = metric(features_prediction_normal, features_target_normal)
    assert diff <= 3.0, \
        f'For same distributions GS should be small, got {diff}'


@pytest.mark.skipif('gudhi' not in sys.modules, reason="Requires gudhi library")
def test_GS_differs_for_not_simular_distributions(
        features_prediction_beta: torch.Tensor, features_target_normal: torch.Tensor) -> None:
    metric = GS(sample_size=64, num_iters=500, i_max=100, num_workers=4)
    diff = metric(features_prediction_beta, features_target_normal)
    assert diff >= 1.0, \
        f'For different distributions GS diff should be big, got {diff}'
