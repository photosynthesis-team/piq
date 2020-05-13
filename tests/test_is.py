from typing import Optional

import torch
import pytest
import torchvision
import numpy as np
from scipy.stats import entropy

from photosynthesis_metrics import IS


# Same as in https://github.com/sbarratt/inception-score-pytorch
# Close to https://github.com/tsc2017/Inception-Score
def logits_to_score_scipy(logits, num_splits=10):
    N = logits.size(0)
    probas = torch.nn.functional.softmax(logits).cpu().numpy()
    split_scores = []
    for i in range(num_splits):
        part = probas[i * (N // num_splits): (i+1) * (N // num_splits), :]
        p_y = np.mean(part, axis=0)
        scores = []
        for k in range(part.shape[0]):
            p_yx = part[k, :]
            scores.append(entropy(p_yx, p_y))
        split_scores.append(np.exp(np.mean(scores)))
    return np.mean(split_scores)


@pytest.fixture(scope='module')
def features_target_normal() -> torch.Tensor:
    m = torch.distributions.Normal(0, 5)
    return m.sample((1000, 20))


@pytest.fixture(scope='module')
def features_prediction_normal() -> torch.Tensor:
    m = torch.distributions.Normal(0, 5)
    return m.sample((1000, 20))


@pytest.fixture(scope='module')
def features_prediction_beta() -> torch.Tensor:
    m = torch.distributions.Beta(2, 2)
    return m.sample((1000, 20))


# @pytest.fixture(scope='module')
# def features_prediction_constant() -> torch.Tensor:
#     return torch.ones(1000, 20)


def test_IS_init() -> None:
    try:
        metric = IS()
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_IS_forward(
    features_target_normal: torch.Tensor, features_prediction_normal: torch.Tensor,) -> None:
    try:
        metric = IS()
        score = metric(features_target_normal, features_prediction_normal)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_IS_returns_variance(
    features_target_normal: torch.Tensor, features_prediction_normal: torch.Tensor) -> None:
    metric = IS(ret_var=True)
    result = metric(features_target_normal, features_prediction_normal)
    assert len(result) == 2, \
        f'Expected to get score and variance, got {result}'
    

def test_IS_returns_target_score(
    features_target_normal: torch.Tensor, features_prediction_normal: torch.Tensor) -> None:
    metric = IS(ret_var=False, ret_target=True)
    result = metric(features_target_normal, features_prediction_normal)
    assert len(result) == 2, \
        f'Expected to get score for prediction and target, got {result}'


def test_IS_similar_for_same_distribution(
    features_target_normal: torch.Tensor, features_prediction_normal: torch.Tensor) -> None:
    metric = IS(ret_var=False, ret_target=True)
    prediction_score, target_score = metric(features_prediction_normal, features_target_normal)
    diff = abs(prediction_score - target_score)
    assert diff <= 1.0, \
        f'For same distributions IS difference should be small, got {diff}'


def test_IS_differs_for_notsimular_distributions(
    features_prediction_beta: torch.Tensor, features_target_normal: torch.Tensor) -> None:
    metric = IS(ret_var=False, ret_target=True)
    prediction_score, target_score = metric(features_prediction_beta, features_target_normal)
    diff = abs(prediction_score - target_score)
    assert diff >= 5.0, \
        f'For different distributions IS diff should be big, got {diff}'

def test_IS_equal_to_scipy_version() -> None:
    prediction_features = torch.distributions.Normal(0, 5).sample((1000, 20))
    metric = IS()
    prediction_score = metric(prediction_features, prediction_features)
    prediction_score_scipy = torch.tensor(logits_to_score_scipy(prediction_features))
    diff = abs(prediction_score - prediction_score_scipy) 
    assert diff <= 1e-4, \
        f'PyTorch and Scipy implementation should match, got diff {diff}'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CPU inference takes ~30 minutes.')
def test_IS_on_CIFAR10_train_equals_to_paper_value() -> None:
    cifar10 = torchvision.datasets.CIFAR10(
        root="downloads/", 
        download=True,
        train=True, 
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]))

    loader = torch.utils.data.DataLoader(cifar10, batch_size=100, num_workers=6)

    model = torchvision.models.inception_v3(pretrained=True, transform_input=False).cuda()
    model.eval()

    target_features = []
    with torch.no_grad():
        upsample = torch.nn.Upsample(size=(299, 299), mode='bilinear')
        for i, batch in enumerate(loader):
            images, labels = batch
            output = model(upsample(images).cuda())
            target_features.append(output)
            # Take only 10000 images, to make everything faster
            if i == 100:
                break

    metric = IS(ret_var=True)
    target_features = torch.cat(target_features, dim=0)
    mean, variance = metric(target_features, target_features)
    # # Values from paper https://arxiv.org/pdf/1801.01973.pdf
    # # CIFAR10 train: 9.737±0.148
    mean_diff, var_diff = abs(mean - 9.737), abs(variance - 0.148)
    assert (mean_diff <= 1.0) and (var_diff <= 0.5), \
        f'Mean and var not close to paper. Mean diff: {mean_diff}, var diff: {var_diff}'


