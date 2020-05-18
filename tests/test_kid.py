import pytest
import torch

from photosynthesis_metrics import KID


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


def test_KID_init() -> None:
    try:
        KID()
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_KID_forward(features_target_normal: torch.Tensor, features_prediction_normal: torch.Tensor, ) -> None:
    try:
        metric = KID()
        metric(features_target_normal, features_prediction_normal)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_KID_fails_for_different_dimensions(features_target_normal: torch.Tensor) -> None:
    features_prediction_normal = torch.rand(1000, 21)
    metric = KID()
    with pytest.raises(AssertionError):
        metric(features_target_normal, features_prediction_normal)


def test_KID_works_for_different_number_of_images_in_stack(features_target_normal: torch.Tensor) -> None:
    features_prediction_normal = torch.rand(1010, 20)
    metric = KID()
    metric(features_target_normal, features_prediction_normal)


def test_KID_returns_variance(features_target_normal: torch.Tensor, features_prediction_normal: torch.Tensor) -> None:
    metric = KID(ret_var=True)
    result = metric(features_target_normal, features_prediction_normal)
    print(result)
    assert len(result) == 2, \
        f'Expected to get score and variance, got {result}'
