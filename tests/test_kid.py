import pytest
import torch

from photosynthesis_metrics import KID, compute_polynomial_mmd


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


# ================== Test function: `compute_polynomial_mmd` ==================
def test_compute_polynomial_mmd_fails_for_different_dimensions(features_target_normal : torch.Tensor) -> None:
    features_prediction_normal = torch.rand(1000, 21)
    with pytest.raises(ValueError):
        compute_polynomial_mmd(features_target_normal, features_prediction_normal)


def test_compute_polynomial_mmd_fails_for_different_number_of_images_in_stack(features_target_normal : torch.Tensor) -> None:
    features_prediction_normal = torch.rand(1001, 20)
    with pytest.raises(AssertionError):
        compute_polynomial_mmd(features_target_normal, features_prediction_normal)

def test_compute_polynomial_mmd_returns_variance(features_target_normal : torch.Tensor, features_prediction_normal : torch.Tensor) -> None:
    result = compute_polynomial_mmd(features_target_normal, features_prediction_normal, ret_var=True)
    assert len(result) == 2, \
        f'Expected to get score and variance, got {result}'



def test_KID_init() -> None:
    try:
        metric = KID()
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_KID_forward(features_target_normal : torch.Tensor, features_prediction_normal : torch.Tensor,) -> None:
    try:
        metric = KID()
        score = metric(features_target_normal, features_prediction_normal)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")
