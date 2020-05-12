r""" This module implements Geometry Score (GS) in PyTorch.

Implementation is inspired by Valentin Khrulkov's (@KhrulkovV) implementation:
https://github.com/KhrulkovV/geometry-score

See paper for details:
https://arxiv.org/pdf/1802.02664.pdf
"""
from typing import Optional

import torch
import numpy as np
import scipy
# from scipy.spatial.distance import cdist


def relative(intervals: np.ndarray, alpha_max: float, i_max: int = 100) -> np.ndarray:
    r"""
    For a collection of intervals this functions computes
    RLT by formulas (2) and (3) from the paper. This function will be typically called
    on the output of the gudhi persistence_intervals_in_dimension function.

    Args:
      intervals: list of intervals e.g. [[0, 1], [0, 2], [0, np.inf]].
      alpha_max: The maximal persistence value
      i_max: Upper bound on the value of beta_1 to compute.

    Returns:
        rlt: Array of size (i_max, ) containing desired RLT.
    """

    persistence_intervals = []
    # If for some interval we have that it persisted up to np.inf
    # we replace this point with alpha_max.
    for interval in intervals:
        if not np.isinf(interval[1]):
            persistence_intervals.append(list(interval))
        elif np.isinf(interval[1]):
            persistence_intervals.append([interval[0], alpha_max])

    # If there are no intervals in H1 then we always observed 0 holes.
    if len(persistence_intervals) == 0:
        rlt = np.zeros(i_max)
        rlt[0] = 1.0
        return rlt

    persistence_intervals_ext = persistence_intervals + [[0, alpha_max]]
    persistence_intervals_ext = np.array(persistence_intervals_ext)
    persistence_intervals = np.array(persistence_intervals)

    # Change in the value of beta_1 may happen only at the boundary points
    # of the intervals
    switch_points = np.sort(np.unique(persistence_intervals_ext.flatten()))
    rlt = np.zeros(i_max)
    for i in range(switch_points.shape[0] - 1):
        midpoint = (switch_points[i] + switch_points[i + 1]) / 2
        s = 0
        for interval in persistence_intervals:
            # Count how many intervals contain midpoint
            if midpoint >= interval[0] and midpoint < interval[1]:
                s = s + 1
        if (s < i_max):
            rlt[s] += (switch_points[i + 1] - switch_points[i])

    return rlt / alpha_max


def lmrk_table(W: np.ndarray, L: np.ndarray):
    r"""Construct an input for the gudhi.WitnessComplex function.
    Args:
        W: Array of size w x d, containing witnesses
        L: Array of size l x d, containing landmarks

    Returns:
        distances: 3D array of size w x l x 2. It satisfies the property that
            distances[i, :, :] is [idx_i, dists_i], where dists_i are the sorted distances 
            from the i-th witness to each point in L and idx_i are the indices of the corresponding points
            in L, e.g., D[i, :, :] = [[0, 0.1], [1, 0.2], [3, 0.3], [2, 0.4]]
        max_dist: Maximal distance between W and L
    """
    a = scipy.spatial.distance.cdist(W, L)
    max_dist = np.max(a)
    idx = np.argsort(a)
    b = a[np.arange(np.shape(a)[0])[:, np.newaxis], idx]
    distances = np.dstack([idx, b])
    return distances, max_dist


def witness(X, sample_size: int = 64, gamma: Optional[float] = None):
    """Compute the persistence intervals for the dataset X using the witness complex.

    Args:
        X: Array of shape (N_samples, data_dim) representing the dataset.
        sample_size: Number of landmarks to use on each iteration.
        gamma: Parameter determining maximum persistence value. Default is `1.0 / 128 * N_imgs / 5000`

    Returns
        A list of persistence intervals and the maximal persistence value.
    """
    # Install gudhi only if needed
    try:
        import gudhi
    except ImportError as e:
        import six
        error = e.__class__(
            "You are likely missing your GUDHI installation, "
            "you should visit http://gudhi.gforge.inria.fr/python/latest/installation.html "
            "for further instructions.\nIf you use conda, you can use\nconda install -c conda-forge gudhi"
        )
        six.raise_from(error, e)

    N = X.shape[0]
    if gamma is None:
        gamma= 1.0 / 128 * N / 5000

    # Randomly sample `sample_size` points from X
    idx = np.random.choice(N, sample_size)
    landmarks = X[idx]

    distances, max_dist = lmrk_table(W=X, L=landmarks)
    wc = gudhi.WitnessComplex(distances)
    alpha_max = max_dist * gamma
    st = wc.create_simplex_tree(max_alpha_square=alpha_max, limit_dimension=2)

    # This seems to modify the st object
    st.persistence(homology_coeff_field=2)
    intervals = st.persistence_intervals_in_dimension(1)
    return intervals, alpha_max


def rlt(X: np.ndarray, sample_size: int = 64, gamma: Optional[float] = None, i_max: int = 100) -> np.ndarray:
    """Implements Algorithm 1 for one sample of landmarks.
 
    Args:
        X: Array of shape (N_samples, data_dim) representing the dataset.
        sample_size: Number of landmarks to use on each iteration.
        gamma: Parameter determining maximum persistence value. Default is `1.0 / 128 * N_imgs / 5000`
        i_max: Upper bound on the value of beta_1 to compute.
    
    Returns:
      An array of size (i_max, ) containing RLT(i, 1, X, L)
        for randomly sampled landmarks.
    """
    intervals, alpha_max = witness(X, sample_size=sample_size, gamma=gamma)
    result = relative(intervals, alpha_max, i_max=i_max)
    return result


def rlts(X: np.ndarray, 
    sample_size: int = 64, num_iters: int = 1000, gamma: Optional[float] = None, i_max: int = 100) -> np.ndarray:
    r"""Implements Algorithm 1 from the paper.

    Args:
        X: Array of shape (N_samples, data_dim) representing the dataset.
        sample_size: Number of landmarks to use on each iteration.
        num_iters: Number of iterations. 
        gamma: Parameter determining maximum persistence value. Default is `1.0 / 128 * N_imgs / 5000`
        i_max: Upper bound on the value of beta_1 to compute.

    Returns:
        rlts: Array of size (num_iters, i_max) containing RLT(i, 1, X, L) for
            `num_iters` collection of randomly sampled landmarks.
    """
    rlts = np.zeros((num_iters, i_max))
    for i in range(num_iters):
        rlts[i, :] = rlt(X, sample_size, gamma, i_max)
    return rlts


class GS(BaseFeatureMetric):
    r"""Interface of Geometry Score.
    It's computed for a whole set of data and can use features from encoder instead of images itself to decrease
    computation cost. GS can compare two data distributions with different number of samples.
    But dimensionalities should match, otherwise it won't be possible to correctly compute statistics.

    Args:
        predicted_features (torch.Tensor): Low-dimension representation of predicted image set. Shape (N_pred, encoder_dim)
        target_features (torch.Tensor): Low-dimension representation of target image set. Shape (N_targ, encoder_dim)

    Returns:
        score (torch.Tensor): Scalar value of the distance between image sets.

    References:
        .. [1] Khrulkov V., Oseledets I. (2018).
        Geometry score: A method for comparing generative adversarial networks.
        arXiv preprint, 2018.
        https://arxiv.org/abs/1802.02664
    """
    def __init__(
        self, sample_size: int = 64, num_iters: int = 10000, gamma: Optional[float] = None, i_max: int = 100) -> None:
        r"""
        Args:
            sample_size: Number of landmarks to use on each iteration. Higher values can give better accuracy,
                but increase computation cost.
            num_iters: Number of iterations.  Higher values can reduce variance,
                but increase computation cost.
            gamma: Parameter determining maximum persistence value. Default is `1.0 / 128 * N_imgs / 5000`
            i_max: Upper bound on i in RLT(i, 1, X, L)

        """
        super(GS, self).__init__()

        self.sample_size = sample_size
        self.num_iters = num_iters
        self.gamma = gamma
        self.i_max = i_max

    def compute_metric(self, predicted_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        r"""Implements Algorithm 2 from the paper.

        Args:
            predicted_features: Samples from data distribution.
                Shape (N_samples, data_dim).
            target_features: Samples from data distribution.
                Shape (N_samples, data_dim).

        Returns:
            score: Scalar value of the distance between distributions.
        """

        # GPU -> CPU -> Numpy (Currently only Numpy realization is supported)
        rlts1
        rlts1
        mean_rlt_predicted = np.mean(predicted_features.detach().cpu().numpy(), axis=0)
        mean_rlt_target = np.mean(target_features.detach().cpu().numpy(), axis=0)

        score = np.sum((mean_rlt_predicted - mean_rlt_target) ** 2)
 
        return torch.tensor(score, device=predicted_features.device)
