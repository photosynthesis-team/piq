import pytest
import torch

from piq import KID


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


@pytest.fixture(scope='module')
def features_x_constant() -> torch.Tensor:
    return torch.ones(1000, 20)


def test_initialization() -> None:
    try:
        KID()
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_forward(features_y_normal, features_x_normal, ) -> None:
    try:
        metric = KID()
        metric(features_y_normal, features_x_normal)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def tes_fails_for_different_dimensions(features_y_normal: torch.Tensor) -> None:
    features_x_normal = torch.rand(1000, 21)
    metric = KID()
    with pytest.raises(AssertionError):
        metric(features_y_normal, features_x_normal)


def test_works_for_different_number_of_images_in_stack(features_y_normal) -> None:
    features_x_normal = torch.rand(1010, 20)
    metric = KID()
    metric(features_y_normal, features_x_normal)


def test_returns_variance(features_y_normal, features_x_normal) -> None:
    metric = KID(ret_var=True)
    result = metric(features_y_normal, features_x_normal)
    print(result)
    assert len(result) == 2, \
        f'Expected to get score and variance, got {result}'
