import pytest
import torch
from contextlib import contextmanager

from piq import FID
from piq.feature_extractors import InceptionV3


@contextmanager
def raise_nothing():
    yield


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, input_range=(0.0, 1.0)):
        self.data = torch.FloatTensor(15, 3, 256, 256).uniform_(*input_range)
        self.mask = torch.rand(15, 3, 256, 256)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.mask[index]

        return {'images': x, 'mask': y}

    def __len__(self):
        return len(self.data)


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


# ================== Test class: `FID` ==================
def test_initialization() -> None:
    FID()


def test_forward(features_y_normal, features_x_normal, device: str) -> None:
    fid = FID()
    fid(features_y_normal.to(device), features_x_normal.to(device))


def test_compute_feats(device: str) -> None:
    dataset = TestDataset()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=3,
        num_workers=2,
    )
    fid = FID()
    model = InceptionV3()
    fid.compute_feats(loader, model, device=device)


@pytest.mark.parametrize("input_range,normalize_input,expectation",
                         [
                             ((0.0, 1.0), True, raise_nothing()),
                             ((-1.0, 1.0), False, raise_nothing()),
                             ((-1.0, 1.0), True, pytest.raises(AssertionError)),
                             ((-10.0, 10.0), False, pytest.raises(AssertionError))
                         ])
def test_inception_input_range(input_range, normalize_input, expectation) -> None:
    with expectation:
        dataset = TestDataset(input_range)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=3,
            num_workers=2,
        )
        fid = FID()
        model = InceptionV3(normalize_input=normalize_input)
        fid.compute_feats(loader, model, device='cpu')
