"""
PyTorch implementation of Inception score (IS)
Reference:
    Salimans T. et al. Improved techniques for training gans (2016)
    https://arxiv.org/abs/1606.03498
    Shane Barratt et al. A Note on the Inception Score
    https://arxiv.org/pdf/1801.01973.pdf

Credits:
    https://github.com/sbarratt/inception-score-pytorch
    https://github.com/tsc2017/Inception-Score
    https://github.com/openai/improved-gan/issues/29
"""
import torch
import torch.nn.functional as F

from piq.base import BaseFeatureMetric
from piq.utils import _validate_input


def inception_score(features: torch.Tensor, num_splits: int = 10):
    r"""Compute Inception Score for a list of image features.
    Expects raw logits from Inception-V3 as input.

    Args:
        features (torch.Tensor): Low-dimension representation of image set. Shape (N_samples, encoder_dim).
        num_splits: Number of parts to divide features. Inception Score is computed for them separately and
            results are then averaged.

    Returns:
        score

        variance

    References:
        "A Note on the Inception Score"
        https://arxiv.org/pdf/1801.01973.pdf

    """
    assert len(features.shape) == 2, \
        f"Features must have shape (N_samples, encoder_dim), got {features.shape}"
    N = features.size(0)

    # Convert logits to probabilities
    probas = F.softmax(features)

    # In the paper the score is computed for 10 splits of the dataset and then averaged.
    partial_scores = []
    for i in range(num_splits):
        subset = probas[i * (N // num_splits): (i + 1) * (N // num_splits), :]

        # Compute KL divergence
        p_y = torch.mean(subset, dim=0)
        scores = []
        for k in range(subset.shape[0]):
            p_yx = subset[k, :]
            scores.append(F.kl_div(p_y.log(), p_yx, reduction='sum'))

        # Compute exponential of the mean of the KL-divergence for each split
        partial_scores.append(torch.tensor(scores).mean().exp())

    partial_scores = torch.tensor(partial_scores)
    return torch.mean(partial_scores).to(features), torch.std(partial_scores).to(features)


class IS(BaseFeatureMetric):
    r"""Creates a criterion that measures difference of Inception Score between two datasets.

    IS is computed separately for predicted :math:`x` and target :math:`y` features and expects raw InceptionV3 model
    logits as inputs.

    Args:
        num_splits: Number of parts to divide features.
            IS is computed for them separately and results are then averaged.
        distance: How to measure distance between scores: ``'l1'`` | ``'l2'``. Default: ``'l1'``.

    Examples:
        >>> is_metric = IS()
        >>> x_feats = torch.rand(10000, 1024)
        >>> y_feats = torch.rand(10000, 1024)
        >>> is: torch.Tensor = is_metric(x_feats, y_feats)

    References:
        "A Note on the Inception Score" https://arxiv.org/pdf/1801.01973.pdf
    """
    def __init__(self, num_splits: int = 10, distance: str = 'l1') -> None:
        r"""

        """
        super(IS, self).__init__()
        self.num_splits = num_splits
        self.distance = distance

    def compute_metric(self, x_features: torch.Tensor, y_features: torch.Tensor) -> torch.Tensor:
        r"""Compute IS.

        Both features should have shape (N_samples, encoder_dim).

        Args:
            x_features: Samples from data distribution. Shape :math:`(N_x, D)`
            y_features: Samples from data distribution. Shape :math:`(N_y, D)`

        Returns:
            L1 or L2 distance between scores for datasets :math:`x` and :math:`y`.
        """
        _validate_input([x_features, y_features], dim_range=(2, 2), size_range=(0, 2))
        x_is, _ = inception_score(x_features, num_splits=self.num_splits)
        y_is, _ = inception_score(y_features, num_splits=self.num_splits)
        if self.distance == 'l1':
            return torch.dist(x_is, y_is, 1)
        elif self.distance == 'l2':
            return torch.dist(x_is, y_is, 2)
        else:
            raise ValueError("Distance should be one of {`l1`, `l2`}")
