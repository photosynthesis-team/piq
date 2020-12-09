import pytest
import torch

from piq import FID
from piq.feature_extractors import InceptionV3


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
def test_initialization() -> None:
    try:
        FID()
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_forward(features_target_normal: torch.Tensor, features_prediction_normal: torch.Tensor, ) -> None:
    try:
        fid = FID()
        fid(features_target_normal, features_prediction_normal)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_compute_feats_cpu() -> None:
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
def test_compute_feats_cuda() -> None:
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


@pytest.mark.parametrize("input_range,normalize_input,is_correct",
                         [
                             ((0.0, 1.0), True, True),
                             ((-1.0, 1.0), False, True),
                             ((-1.0, 1.0), True, False),
                             ((-10.0, 10.0), False, False)
                         ])
def test_inception_input_range(input_range, normalize_input, is_correct) -> None:
    try:
        dataset = TestDataset(input_range)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=3,
            num_workers=2,
        )
        fid = FID()
        model = InceptionV3(normalize_input=normalize_input)
        if is_correct:
            fid._compute_feats(loader, model, device='cpu')
        else:
            with pytest.raises(Exception):
                fid._compute_feats(loader, model, device='cpu')
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")
