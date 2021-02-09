import pytest
import torch

from piq import GS


@pytest.fixture(scope='module')
def features_y_normal() -> torch.Tensor:
    return torch.rand(1000, 20)


@pytest.fixture(scope='module')
def features_x_normal() -> torch.Tensor:
    return torch.rand(1000, 20)


@pytest.fixture(scope='module')
def features_x_beta() -> torch.Tensor:
    m = torch.distributions.Beta(torch.FloatTensor([2]), torch.FloatTensor([2]))
    return m.sample([1000, 20]).squeeze()


# ================== Test class: `GS` ==================
def test_initialization() -> None:
    try:
        GS()
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_forward(features_y_normal, features_x_normal,) -> None:
    try:
        metric = GS(num_iters=10, sample_size=8)
        metric(features_y_normal, features_x_normal)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


@pytest.mark.skip(reason="Randomnly fails, fix in separate PR")
def test_similar_for_same_distribution(features_y_normal, features_x_normal) -> None:
    metric = GS(sample_size=1000, num_iters=100, i_max=1000, num_workers=4)
    diff = metric(features_x_normal, features_y_normal)
    assert diff <= 2.0, \
        f'For same distributions GS should be small, got {diff}'


@pytest.mark.skip(reason="Randomnly fails, fix in separate PR")
def test_differs_for_not_simular_distributions(features_x_beta, features_y_normal) -> None:
    metric = GS(sample_size=1000, num_iters=100, i_max=1000, num_workers=4)
    diff = metric(features_x_beta, features_y_normal)
    assert diff >= 5.0, \
        f'For different distributions GS diff should be big, got {diff}'
