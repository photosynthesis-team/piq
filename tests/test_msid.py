import torch
import pytest

from photosynthesis_metrics import MSID
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
def prediction() -> torch.Tensor:
    return torch.rand(3, 3, 256, 256)


@pytest.fixture(scope='module')
def target() -> torch.Tensor:
    return torch.rand(3, 3, 256, 256)


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


# ================== Test class: `MSID` ==================
def test_MSID_fails_for_different_dimensions(features_target_normal: torch.Tensor) -> None:
    features_prediction_normal = torch.rand(1000, 21)
    metric = MSID()
    with pytest.raises(AssertionError):
        metric(features_target_normal, features_prediction_normal)


def test_compute_msid_works_for_different_number_of_images_in_stack(features_target_normal: torch.Tensor) -> None:
    features_prediction_normal = torch.rand(1001, 20)
    metric = MSID()
    try:
        metric(features_target_normal, features_prediction_normal)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_MSID_init() -> None:
    try:
        MSID()
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


@pytest.mark.skip(reason="Sometimes it doesn't work.")
def test_msid_is_smaller_for_equal_tensors(
        features_target_normal: torch.Tensor,
        features_prediction_normal: torch.Tensor,
        features_prediction_constant: torch.Tensor
) -> None:
    metric = MSID()
    measure = metric(features_target_normal, features_prediction_normal)
    measure_constant = metric(features_target_normal, features_prediction_normal)
    assert measure <= measure_constant, \
        f'MSID should be smaller for samples from the same distribution, got {measure} and {measure_constant}'


def test_MSID_forward(features_target_normal: torch.Tensor, features_prediction_normal: torch.Tensor, ) -> None:
    try:
        metric = MSID()
        metric(features_target_normal, features_prediction_normal)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_MSID_compute_feats_cpu() -> None:
    try:
        dataset = TestDataset()
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=3,
            num_workers=2,
        )
        metric = MSID()
        model = InceptionV3()
        metric._compute_feats(loader, model, device='cpu')
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason='No need to run test on GPU if there is no GPU.')
def test_MSID_compute_feats_cuda() -> None:
    try:
        dataset = TestDataset()
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=3,
            num_workers=2,
        )
        metric = MSID()
        model = InceptionV3()
        metric._compute_feats(loader, model, device='cuda')
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")
