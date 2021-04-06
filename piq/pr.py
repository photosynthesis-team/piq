"""
PyTorch implementation of Improved Precision and Recall (P&R)
Reference:
    Kynkäänniemi T. et al. Improved Precision and Recall Metric for Assessing Generative Models,
    https://arxiv.org/abs/1904.06991
Credits:
    https://github.com/clovaai/generative-evaluation-prdc/blob/master/prdc/prdc.py
"""

from typing import Tuple
import torch

from piq.base import BaseFeatureMetric
from piq.utils import _validate_input


def _compute_pairwise_distance(data_x: torch.Tensor, data_y: torch.Tensor = None) -> torch.Tensor:
    """Compute Euclidean distance between x and y
    Args:
        data_x: Tensor of shape (N, feature_dim)
        data_y: Tensor of shape (N, feature_dim)
    Returns:
        Tensor of shape (N, N) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = torch.cdist(data_x, data_y, p=2)
    return dists


def _get_kth_value(unsorted: torch.Tensor, k: int, axis: int = -1) -> torch.Tensor:
    """
    Args:
        unsorted: Tensor of any dimensionality.
        k: Int of the k-th value to retrieve.
    Returns:
        kth values along the designated axis.
    """
    k_smallests = torch.topk(unsorted, k, dim=axis, largest=False)[0]
    kth_values = k_smallests.max(dim=axis)[0]
    return kth_values


def _compute_nearest_neighbour_distances(input_features: torch.Tensor, nearest_k: int) -> torch.Tensor:
    """
    Compute K-nearest neighbour distances.
    Args:
        input_features: Tensor of shape (N, feature_dim)
        nearest_k: Int of the k-th nearest neighbour.
    Returns:
        Distances to kth nearest neighbours.
    """
    distances = _compute_pairwise_distance(input_features)
    radii = _get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


class PR(BaseFeatureMetric):
    r"""
    Interface of Improved Precision and Recall.
    It's computed for a whole set of data and uses features from encoder instead of images itself to decrease
    computation cost. Precision and Recall can compare two data distributions with different number of samples.
    But dimensionalities should match, otherwise it won't be possible to correctly compute statistics.

    Args:
        real_features: Samples from data distribution. Shape :math:`(N_x, D)`
        fake_features: Samples from generated distribution. Shape :math:`(N_y, D)`

    Returns:
        precision: Scalar value of the precision of image sets features.
        recall: Scalar value of the recall of image sets features.

    References:
        .. [1] Kynkäänniemi T. et al. (2019).
           Improved Precision and Recall Metric for Assessing Generative Models.
           Advances in Neural Information Processing Systems,
           https://arxiv.org/abs/1904.06991
    """

    def compute_metric(self, real_features: torch.Tensor, fake_features: torch.Tensor, nearest_k: int = 5)\
            -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Creates non-parametric representations of the manifolds of real and generated data and computes the precision
        and recall between them.
        Args:
            real_features: Samples from data distribution. Shape :math:`(N_x, D)`
            fake_features: Samples from fake distribution. Shape :math:`(N_x, D)`
            nearest_k: Nearest neighbor to compute the non-parametric representation. Shape :math:`1`
        Returns:
            Precision and recall.
        """
        _validate_input([real_features, fake_features], dim_range=(2, 2), size_range=(1, 2))
        real_nearest_neighbour_distances = _compute_nearest_neighbour_distances(real_features, nearest_k)
        fake_nearest_neighbour_distances = _compute_nearest_neighbour_distances(fake_features, nearest_k)
        distance_real_fake = _compute_pairwise_distance(real_features, fake_features)

        precision = (
                distance_real_fake < real_nearest_neighbour_distances.unsqueeze(1)
        ).any(dim=0).float().mean()

        recall = (
                distance_real_fake < fake_nearest_neighbour_distances.unsqueeze(0)
        ).any(dim=1).float().mean()

        return precision, recall
