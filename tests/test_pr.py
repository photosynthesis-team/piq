import pytest
import torch

from piq import PR


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


def test_forward(features_y_normal, features_x_normal, ) -> None:
    metric = PR()
    metric(features_y_normal, features_x_normal)


def test_fails_for_different_dimensions(features_y_normal: torch.Tensor) -> None:
    features_x_normal = torch.rand(1000, 21)
    metric = PR()
    with pytest.raises(AssertionError):
        metric(features_y_normal, features_x_normal)


def test_works_for_different_number_of_images_in_stack(features_y_normal) -> None:
    features_x_normal = torch.rand(1010, 20)
    metric = PR()
    metric(features_y_normal, features_x_normal)


def test_return_two_scores(features_y_normal, features_x_normal) -> None:
    metric = PR()
    result = metric(features_y_normal, features_x_normal)
    print(result)
    assert len(result) == 2, \
        f'Expected to get precision and recall, got {result}'


def test_ones_on_same_features(features_y_normal) -> None:
    features_x_normal = features_y_normal
    metric = PR()
    precision, recall = metric(features_y_normal, features_x_normal)
    assert torch.isclose(precision, torch.ones_like(precision)) and torch.isclose(recall, torch.ones_like(recall)), \
        f"Expected precision == 1.0 and recall == 1.0, got precision={precision}, recall={recall}"
