r""" This module implements Geometry Score (GS) in PyTorch.
Implementation is inspired by Valentin Khrulkov's (@KhrulkovV) implementation:
https://github.com/KhrulkovV/geometry-score
See paper for details:
https://arxiv.org/pdf/1802.02664.pdf
"""
import torch

import numpy as np

from typing import Optional, Tuple, Union
from multiprocessing import Pool
from warnings import warn

from piq.base import BaseFeatureMetric
from piq.utils import _validate_input, _parse_version


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
        if np.isinf(interval[1]):
            persistence_intervals.append([interval[0], alpha_max])
        else:
            persistence_intervals.append(list(interval))

    # If there are no intervals in H1 then we always observed 0 holes.
    if len(persistence_intervals) == 0:
        rlt = np.zeros(i_max)
        rlt[0] = 1.0
        return rlt

    persistence_intervals_ext = np.array(persistence_intervals + [[0, alpha_max]])
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


def lmrk_table(witnesses: np.ndarray, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""Construct an input for the gudhi.WitnessComplex function.
    Args:
        witnesses: Array with shape (w, d), containing witnesses.
        landmarks: Array with shape (l, d), containing landmarks.
    Returns:
        distances: 3D array with shape (w, l, 2). It satisfies the property that
            distances[i, :, :] is [idx_i, dists_i], where dists_i are the sorted distances
            from the i-th witness to each point in L and idx_i are the indices of the corresponding points
            in L, e.g., D[i, :, :] = [[0, 0.1], [1, 0.2], [3, 0.3], [2, 0.4]]
        max_dist: Maximal distance between W and L
    """
    try:
        import scipy
    except ImportError:
        raise ImportError("Scipy is required for computation of the Geometry Score but not installed. "
                          "Please install scipy using the following command: pip install --user scipy")

    recommended_scipy_version = _parse_version("1.3.3")
    scipy_version = _parse_version(scipy.__version__)
    if len(scipy_version) != 0 and scipy_version < recommended_scipy_version:
        warn(f'Scipy of version {scipy.__version__} is used while version >= {recommended_scipy_version} is '
             f'recommended. Consider updating scipy to avoid potential long compute time with older versions.')

    from scipy.spatial.distance import cdist

    a = cdist(witnesses, landmarks)
    max_dist = np.max(a)
    idx = np.argsort(a)
    b = a[np.arange(np.shape(a)[0])[:, np.newaxis], idx]
    distances = np.dstack([idx, b])
    return distances, max_dist


def witness(features: np.ndarray, sample_size: int = 64, gamma: Optional[float] = None) \
        -> Tuple[np.ndarray, np.ndarray]:
    """Compute the persistence intervals for the dataset of features using the witness complex.

    Args:
        features: Array with shape (N_samples, data_dim) representing the dataset.
        sample_size: Number of landmarks to use on each iteration.
        gamma: Parameter determining maximum persistence value. Default is `1.0 / 128 * N_imgs / 5000`

    Returns
        A list of persistence intervals and the maximal persistence value.
    """
    try:
        import gudhi
    except ImportError:
        raise ImportError("GUDHI is required for computation of the Geometry Score but not installed. "
                          "Please install scipy using the following command: pip install --user gudhi")

    recommended_gudhi_version = _parse_version("3.2.0")
    gudhi_version = _parse_version(gudhi.__version__)
    if len(gudhi_version) != 0 and gudhi_version < recommended_gudhi_version:
        warn(f'GUDHI of version {gudhi.__version__} is used while version >= {recommended_gudhi_version} is '
             f'recommended. Consider updating GUDHI to avoid potential problems.')

    N = features.shape[0]
    if gamma is None:
        gamma = 1.0 / 128 * N / 5000

    # Randomly sample `sample_size` points from X
    np.random.seed()
    idx = np.random.choice(N, sample_size)
    landmarks = features[idx]

    distances, max_dist = lmrk_table(witnesses=features, landmarks=landmarks)
    wc = gudhi.WitnessComplex(distances)
    alpha_max = max_dist * gamma
    st = wc.create_simplex_tree(max_alpha_square=alpha_max, limit_dimension=2)

    # This seems to modify the st object
    st.persistence(homology_coeff_field=2)
    intervals = st.persistence_intervals_in_dimension(1)
    return intervals, alpha_max


class GS(BaseFeatureMetric):
    r"""Interface of Geometry Score.
    It's computed for a whole set of data and can use features from encoder instead of images itself to decrease
    computation cost. GS can compare two data distributions with different number of samples.
    Dimensionalities of features should match, otherwise it won't be possible to correctly compute statistics.

    Args:
        sample_size: Number of landmarks to use on each iteration.
            Higher values can give better accuracy, but increase computation cost.
        num_iters: Number of iterations.
            Higher values can reduce variance, but increase computation cost.
        gamma: Parameter determining maximum persistence value. Default is ``1.0 / 128 * N_imgs / 5000``
        i_max: Upper bound on i in RLT(i, 1, X, L)
        num_workers: Number of processes used for GS computation.

    Examples:
        >>> gs_metric = GS()
        >>> x_feats = torch.rand(10000, 1024)
        >>> y_feats = torch.rand(10000, 1024)
        >>> gs: torch.Tensor = gs_metric(x_feats, y_feats)

    References:
        Khrulkov V., Oseledets I. (2018).
        Geometry score: A method for comparing generative adversarial networks.
        arXiv preprint, 2018.
        https://arxiv.org/abs/1802.02664

    Note:
        Computation is heavily CPU dependent, adjust ``num_workers`` parameter according to your system configuration.
        GS metric requiers ``gudhi`` library which is not installed by default.
        For conda, write: ``conda install -c conda-forge gudhi``,
        otherwise follow installation guide: http://gudhi.gforge.inria.fr/python/latest/installation.html
    """

    def __init__(self, sample_size: int = 64, num_iters: int = 1000, gamma: Optional[float] = None,
                 i_max: int = 100, num_workers: int = 4) -> None:
        super().__init__()
        self.sample_size = sample_size
        self.num_iters = num_iters
        self.gamma = gamma
        self.i_max = i_max
        self.num_workers = num_workers

    def compute_metric(self, x_features: torch.Tensor, y_features: torch.Tensor) -> torch.Tensor:
        r"""Implements Algorithm 2 from the paper.

        Args:
            x_features: Samples from data distribution. Shape :math:`(N_x, D)`
            y_features: Samples from data distribution. Shape :math:`(N_y, D)`

        Returns:
            Scalar value of the distance between distributions.
        """
        _validate_input([x_features, y_features], dim_range=(2, 2), size_range=(1, 2))
        with Pool(self.num_workers) as p:
            self.features = x_features.detach().cpu().numpy()
            pool_results = p.map(self._relative_living_times, range(self.num_iters))
            mean_rlt_x = np.vstack(pool_results).mean(axis=0)

            self.features = y_features.detach().cpu().numpy()
            pool_results = p.map(self._relative_living_times, range(self.num_iters))
            mean_rlt_y = np.vstack(pool_results).mean(axis=0)

        score = np.sum((mean_rlt_x - mean_rlt_y) ** 2)

        return torch.tensor(score, device=x_features.device) * 1000

    def _relative_living_times(self, idx: int) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        r"""Implements Algorithm 1 for two samples of landmarks.

        Args:
            idx : Dummy argument. Used for multiprocessing.Pool to work correctly

        Returns:
            An array of size (i_max, ) containing RLT(i, 1, X, L)
            for randomly sampled landmarks.
        """
        intervals, alpha_max = witness(self.features, sample_size=self.sample_size, gamma=self.gamma)
        rlt = relative(intervals, alpha_max, i_max=self.i_max)
        return rlt
