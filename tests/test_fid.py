import pytest
import torch

from photosynthesis_metrics import FID
from photosynthesis_metrics.feature_extractors.fid_inception import InceptionV3


class TestDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = torch.randn(15, 3, 256, 256)
        self.mask = torch.randn(15, 3, 256, 256)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.mask[index]

        return {'images': x, 'mask': y}

    def __len__(self):
        return len(self.data)


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


# ================== Test class: `FID` ==================
def test_FID_init() -> None:
    try:
        FID()
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_FID_forward(features_target_normal: torch.Tensor, features_prediction_normal: torch.Tensor, ) -> None:
    try:
        fid = FID()
        fid(features_target_normal, features_prediction_normal)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_FID_compute_feats_cpu() -> None:
    try:
        dataset = TestDataset()
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=3,
            num_workers=2,
        )
        fid = FID()
        model = InceptionV3()
        fid._compute_feats(loader, model, device='cpu')
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test on GPU if there is no GPU.')
def test_FID_compute_feats_cuda() -> None:
    try:
        dataset = TestDataset()
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=3,
            num_workers=2,
        )
        fid = FID()
        model = InceptionV3()
        fid._compute_feats(loader, model, device='cuda')
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")
