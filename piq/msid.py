r"""Implementation of Multi-scale Evaluation metric, based on paper
 https://arxiv.org/abs/1905.11141 and author's repository https://github.com/xgfs/msid
"""
from typing import List, Tuple, Optional
from warnings import warn

import torch
import numpy as np

from piq.base import BaseFeatureMetric
from piq.utils import _validate_input, _parse_version


EPSILON = 1e-6
NORMALIZATION = 1e6


def _np_euc_cdist(data: np.ndarray) -> np.ndarray:
    dd = np.sum(data * data, axis=1)
    dist = -2 * np.dot(data, data.T)
    dist += dd + dd[:, np.newaxis]
    np.fill_diagonal(dist, 0)
    np.sqrt(dist, dist)
    return dist


def _construct_graph_sparse(data: np.ndarray, k: int) -> np.ndarray:
    n = len(data)
    from scipy.sparse import lil_matrix
    spmat = lil_matrix((n, n))
    dd = np.sum(data * data, axis=1)

    for i in range(n):
        dists = dd - 2 * data[i, :].dot(data.T)
        inds = np.argpartition(dists, k + 1)[:k + 1]
        inds = inds[inds != i]
        spmat[i, inds] = 1

    return spmat.tocsr()


def _laplacian_sparse(matrix: np.ndarray, normalized: bool = True) -> np.ndarray:
    from scipy.sparse import diags, eye
    row_sum = matrix.sum(1).A1
    if not normalized:
        return diags(row_sum) - matrix

    row_sum_sqrt = diags(1 / np.sqrt(row_sum))
    return eye(matrix.shape[0]) - row_sum_sqrt.dot(matrix).dot(row_sum_sqrt)


def _lanczos_m(A: np.ndarray, m: int, nv: int, rademacher: bool, starting_vectors: Optional[np.ndarray] = None) \
        -> Tuple[np.ndarray, np.ndarray]:
    r"""Lanczos algorithm computes symmetric m x m tridiagonal matrix T and matrix V with orthogonal rows
        constituting the basis of the Krylov subspace K_m(A, x),
        where x is an arbitrary starting unit vector.
        This implementation parallelizes `nv` starting vectors.

    Args:
        A: matrix based on which the Krylov subspace will be built.
        m: Number of Lanczos steps.
        nv: Number of random vectors.
        rademacher: True to use Rademacher distribution,
            False - standard normal for random vectors
        starting_vectors: Specified starting vectors.

    Returns:
        T: Array with shape (nv, m, m), where T[i, :, :] is the i-th symmetric tridiagonal matrix.
        V: Array with shape (n, m, nv) where, V[:, :, i] is the i-th matrix with orthogonal rows.
    """
    orthtol = 1e-5
    if starting_vectors is None:
        if rademacher:
            starting_vectors = np.sign(np.random.randn(A.shape[0], nv))
        else:
            starting_vectors = np.random.randn(A.shape[0], nv)  # init random vectors in columns: n x nv
    V = np.zeros((starting_vectors.shape[0], m, nv))
    T = np.zeros((nv, m, m))

    np.divide(starting_vectors, np.linalg.norm(starting_vectors, axis=0), out=starting_vectors)  # normalize each column
    V[:, 0, :] = starting_vectors

    w = A.dot(starting_vectors)
    alpha = np.einsum('ij,ij->j', w, starting_vectors)
    w -= alpha[None, :] * starting_vectors
    beta = np.einsum('ij,ij->j', w, w)
    np.sqrt(beta, beta)

    T[:, 0, 0] = alpha
    T[:, 0, 1] = beta
    T[:, 1, 0] = beta

    np.divide(w, beta[None, :], out=w)
    V[:, 1, :] = w
    t = np.zeros((m, nv))

    for i in range(1, m):
        old_starting_vectors = V[:, i - 1, :]
        starting_vectors = V[:, i, :]

        w = A.dot(starting_vectors)  # sparse @ dense
        w -= beta[None, :] * old_starting_vectors  # n x nv
        np.einsum('ij,ij->j', w, starting_vectors, out=alpha)

        T[:, i, i] = alpha

        if i < m - 1:
            w -= alpha[None, :] * starting_vectors  # n x nv

            # reortho
            np.einsum('ijk,ik->jk', V, w, out=t)
            w -= np.einsum('ijk,jk->ik', V, t)
            np.einsum('ij,ij->j', w, w, out=beta)
            np.sqrt(beta, beta)
            np.divide(w, beta[None, :], out=w)

            T[:, i, i + 1] = beta
            T[:, i + 1, i] = beta

            # more reotho
            innerprod = np.einsum('ijk,ik->jk', V, w)
            reortho = False
            for _ in range(100):
                if not (innerprod > orthtol).sum():
                    reortho = True
                    break

                np.einsum('ijk,ik->jk', V, w, out=t)
                w -= np.einsum('ijk,jk->ik', V, t)
                np.divide(w, np.linalg.norm(w, axis=0)[None, :], out=w)
                innerprod = np.einsum('ijk,ik->jk', V, w)

            V[:, i + 1, :] = w

            if (np.abs(beta) > 1e-6).sum() == 0 or not reortho:
                break

    return T, V


def _slq(A: np.ndarray, m: int, niters: int, rademacher: bool) -> np.ndarray:
    r"""Compute the trace of matrix exponential

    Args:
        A: Square matrix in trace(exp(A)).
        m: Number of Lanczos steps.
        niters: Number of quadratures (also, the number of random vectors in the hutchinson trace estimator).
        rademacher: True to use Rademacher distribution,
            False - standard normal for random vectors in Hutchinson.
    Returns:
        trace: Estimate of trace of matrix exponential.
    """
    T, _ = _lanczos_m(A, m, niters, rademacher)
    eigvals, eigvecs = np.linalg.eigh(T)
    expeig = np.exp(eigvals)
    sqeigv1 = np.power(eigvecs[:, 0, :], 2)
    trace = A.shape[-1] * (expeig * sqeigv1).sum() / niters
    return trace


def _slq_ts(A: np.ndarray, m: int, niters: int, ts: np.ndarray, rademacher: bool) -> np.ndarray:
    r"""Compute the trace of matrix exponential

    Args:
        A: Square matrix in trace(exp(-t*A)), where t is temperature
        m: Number of Lanczos steps.
        niters: Number of quadratures (also, the number of random vectors in the hutchinson trace estimator).
        ts: Array with temperatures.
        rademacher: True to use Rademacher distribution, False - standard normal for random vectors in Hutchinson

    Returns:
        trace: Estimate of trace of matrix exponential across temperatures `ts`
    """
    T, _ = _lanczos_m(A, m, niters, rademacher)
    eigvals, eigvecs = np.linalg.eigh(T)
    expeig = np.exp(-np.outer(ts, eigvals)).reshape(ts.shape[0], niters, m)
    sqeigv1 = np.power(eigvecs[:, 0, :], 2)
    traces = A.shape[-1] * (expeig * sqeigv1).sum(-1).mean(-1)
    return traces


def _slq_ts_fs(A: np.ndarray, m: int, niters: int, ts: np.ndarray, rademacher: bool, fs: List) -> np.ndarray:
    r"""Compute the trace of matrix functions

    Args:
        A: Square matrix in trace(exp(-t*A)), where t is temperature.
        m: Number of Lanczos steps.
        niters: Number of quadratures (also, the number of random vectors in the hutchinson trace estimator).
        ts: Array with temperatures.
        rademacher: True to use Rademacher distribution, else - standard normal for random vectors in Hutchinson
        fs: A list of functions.

    Returns:
        traces: Estimate of traces for each of the functions in `fs`.
    """
    T, _ = _lanczos_m(A, m, niters, rademacher)
    eigvals, eigvecs = np.linalg.eigh(T)
    traces = np.zeros((len(fs), len(ts)))
    for i, f in enumerate(fs):
        expeig = f(-np.outer(ts, eigvals)).reshape(ts.shape[0], niters, m)
        sqeigv1 = np.power(eigvecs[:, 0, :], 2)
        traces[i, :] = A.shape[-1] * (expeig * sqeigv1).sum(-1).mean(-1)
    return traces


def _slq_red_var(A: np.ndarray, m: int, niters: int, ts: np.ndarray, rademacher: bool) -> np.ndarray:
    r"""Compute the trace of matrix exponential with reduced variance

    Args:
        A: Square matrix in trace(exp(-t*A)), where t is temperature.
        m: Number of Lanczos steps.
        niters: Number of quadratures (also, the number of random vectors in the hutchinson trace estimator).
        ts: Array with temperatures.

    Returns:
        traces: Estimate of trace for each temperature value in `ts`.
    """
    fs = [np.exp, lambda x: x]

    traces = _slq_ts_fs(A, m, niters, ts, rademacher, fs)
    subee = traces[0, :] - traces[1, :] / np.exp(ts)
    sub = - ts * A.shape[0] / np.exp(ts)
    return subee + sub


def _build_graph(data: np.ndarray, k: int = 5, normalized: bool = True):
    r"""Return Laplacian from data or load preconstructed from path

    Args:
        data: Samples.
        k: Number of neighbours for graph construction.
        normalized: if True, use nnormalized Laplacian.

    Returns:
        L: Laplacian of the graph constructed with data.
    """

    A = _construct_graph_sparse(data, k)
    A = (A + A.T) / 2
    A.data = np.ones(A.data.shape)
    L = _laplacian_sparse(A, normalized)
    return L


def _normalize_msid(msid: np.ndarray, normalization: str, n: int, k: int, ts: np.ndarray):
    normed_msid = msid.copy()
    if normalization == 'empty':
        normed_msid /= n
    elif normalization == 'complete':
        normed_msid /= (1 + (n - 1) * np.exp(-(1 + 1 / (n - 1)) * ts))
    elif normalization == 'er':
        xs = np.linspace(0, 1, n)
        er_spectrum = 4 / np.sqrt(k) * xs + 1 - 2 / np.sqrt(k)
        er_msid = np.exp(-np.outer(ts, er_spectrum)).sum(-1)
        normed_msid = normed_msid / (er_msid + EPSILON)
    elif not (normalization == 'none' or normalization is None):
        raise ValueError('Unknown normalization parameter!')

    return normed_msid


def _msid_descriptor(x: np.ndarray, ts: np.ndarray = np.logspace(-1, 1, 256), k: int = 5, m: int = 10,
                     niters: int = 100, rademacher: bool = False, normalized_laplacian: bool = True,
                     normalize: str = 'empty') \
        -> np.ndarray:
    r"""Compute the msid descriptor for a single set of samples

    Args:
        x: Samples from data distribution. Shape (N_samples, data_dim)
        ts: Temperature values.
        k: Number of neighbours for graph construction.
        m: Lanczos steps in SLQ.
        niters: Number of starting random vectors for SLQ.
        rademacher: True to use Rademacher distribution,
            False - standard normal for random vectors in Hutchinson.
        normalized_laplacian: if True, use normalized Laplacian
        normalize: 'empty' for average heat kernel (corresponds to the empty graph normalization of NetLSD),
                'complete' for the complete, 'er' for erdos-renyi normalization, 'none' for no normalization
    Returns:
        normed_msidx: normalized msid descriptor
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

    Lx = _build_graph(x, k, normalized_laplacian)

    nx = Lx.shape[0]
    msidx = _slq_red_var(Lx, m, niters, ts, rademacher)

    normed_msidx = _normalize_msid(msidx, normalize, nx, k, ts) * NORMALIZATION

    return normed_msidx


class MSID(BaseFeatureMetric):
    r"""Creates a criterion that measures MSID score for two batches of images
    It's computed for a whole set of data and uses features from encoder instead of images itself
    to decrease computation cost. MSID can compare two data distributions with different
    number of samples or different dimensionalities.

    Args:
        ts: Temperature values. If ``None``, the default value ``torch.logspace(-1, 1, 256)`` is used.
        k: Number of neighbours for graph construction.
        m: Lanczos steps in SLQ.
        niters: Number of starting random vectors for SLQ.
        rademacher: True to use Rademacher distribution,
            False - standard normal for random vectors in Hutchinson.
        normalized_laplacian: if True, use normalized Laplacian.
        normalize: ``'empty'`` for average heat kernel (corresponds to the empty graph normalization of NetLSD),
            ``'complete'`` for the complete, ``'er'`` for Erdos-Renyi normalization, ``'none'`` for no normalization
        msid_mode: ``'l2'`` to compute the L2 norm of the distance between `msid1` and `msid2`;
            ``'max'`` to find the maximum absolute difference between two descriptors over temperature

    Examples:
        >>> msid_metric = MSID()
        >>> x_feats = torch.rand(10000, 1024)
        >>> y_feats = torch.rand(10000, 1024)
        >>> msid: torch.Tensor = msid_metric(x_feats, y_feats)

    References:
        Tsitsulin, A., Munkhoeva, M., Mottin, D., Karras, P., Bronstein, A., Oseledets, I., & Müller, E. (2019).
        The shape of data: Intrinsic distance for data distributions.
        https://arxiv.org/abs/1905.11141
    """

    def __init__(self, ts: torch.Tensor = None, k: int = 5, m: int = 10, niters: int = 100,
                 rademacher: bool = False, normalized_laplacian: bool = True, normalize: str = 'empty',
                 msid_mode: str = "max") -> None:
        super(MSID, self).__init__()

        if ts is None:
            ts = torch.logspace(-1, 1, 256)

        self.ts = ts.numpy()  # MSID works only with Numpy tensors
        self.k = k
        self.m = m
        self.niters = niters
        self.rademacher = rademacher
        self.msid_mode = msid_mode
        self.normalized_laplacian = normalized_laplacian
        self.normalize = normalize

    def compute_metric(self, x_features: torch.Tensor, y_features: torch.Tensor) -> torch.Tensor:

        r"""Compute MSID score between two sets of samples.

        Args:
            x_features: Samples from data distribution. Shape :math:`(N_x, D_x)`
            y_features: Samples from data distribution. Shape :math:`(N_y, D_y)`

        Returns:
            Scalar value of the distance between distributions.
        """
        _validate_input([x_features, y_features], dim_range=(2, 2), size_range=(1, 2))
        normed_msid_x = _msid_descriptor(
            x_features.detach().cpu().numpy(),
            ts=self.ts,
            k=self.k,
            m=self.m,
            niters=self.niters,
            rademacher=self.rademacher,
            normalized_laplacian=self.normalized_laplacian,
            normalize=self.normalize
        )
        normed_msid_y = _msid_descriptor(
            y_features.detach().cpu().numpy(),
            ts=self.ts,
            k=self.k,
            m=self.m,
            niters=self.niters,
            rademacher=self.rademacher,
            normalized_laplacian=self.normalized_laplacian,
            normalize=self.normalize
        )

        c = np.exp(-2 * (self.ts + 1 / self.ts))
        if self.msid_mode == 'l2':
            score = np.linalg.norm(normed_msid_x - normed_msid_y)
        elif self.msid_mode == 'max':
            score = np.amax(c * np.abs(normed_msid_x - normed_msid_y))
        else:
            raise ValueError('Mode must be in {`l2`, `max`}')

        return torch.tensor(score, device=x_features.device)
