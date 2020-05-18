r"""Implemetation of Multi-scale Evaluation metric, based on paper
 https://arxiv.org/abs/1905.11141 and author's repository https://github.com/xgfs/msid
"""
from typing import List, Tuple

import torch
import numpy as np

from scipy.sparse import lil_matrix, diags, eye

from photosynthesis_metrics.base import BaseFeatureMetric

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
    spmat = lil_matrix((n, n))
    dd = np.sum(data * data, axis=1)

    for i in range(n):
        dists = dd - 2 * data[i, :].dot(data.T)
        inds = np.argpartition(dists, k + 1)[:k + 1]
        inds = inds[inds != i]
        spmat[i, inds] = 1

    return spmat.tocsr()


def _laplacian_sparse(A: np.ndarray, normalized: bool = True):
    D = A.sum(1).A1
    if normalized:
        Dsqrt = diags(1 / np.sqrt(D))
        L = eye(A.shape[0]) - Dsqrt.dot(A).dot(Dsqrt)
    else:
        L = diags(D) - A
    return L


def _lanczos_m(
        A: np.ndarray,
        m: int,
        nv: int,
        rademacher: bool,
        SV: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    r"""Lanczos algorithm computes symmetric m x m tridiagonal matrix T and matrix V with orthogonal rows
        constituting the basis of the Krylov subspace K_m(A, x),
        where x is an arbitrary starting unit vector.
        This implementation parallelizes `nv` starting vectors.

    Args:
        m: Number of Lanczos steps.
        nv: Number of random vectors.
        rademacher: True to use Rademacher distribution,
            False - standard normal for random vectors
        SV: Specified starting vectors.

    Returns:
        T: A nv x m x m tensor, T[i, :, :] is the ith symmetric tridiagonal matrix.
        V: A n x m x nv tensor, V[:, :, i] is the ith matrix with orthogonal rows.
    """
    orthtol = 1e-5
    if type(SV) != np.ndarray:
        if rademacher:
            SV = np.sign(np.random.randn(A.shape[0], nv))
        else:
            SV = np.random.randn(A.shape[0], nv)  # init random vectors in columns: n x nv
    V = np.zeros((SV.shape[0], m, nv))
    T = np.zeros((nv, m, m))

    np.divide(SV, np.linalg.norm(SV, axis=0), out=SV)  # normalize each column
    V[:, 0, :] = SV

    w = A.dot(SV)
    alpha = np.einsum('ij,ij->j', w, SV)
    w -= alpha[None, :] * SV
    beta = np.einsum('ij,ij->j', w, w)
    np.sqrt(beta, beta)

    T[:, 0, 0] = alpha
    T[:, 0, 1] = beta
    T[:, 1, 0] = beta

    np.divide(w, beta[None, :], out=w)
    V[:, 1, :] = w
    t = np.zeros((m, nv))

    for i in range(1, m):
        SVold = V[:, i - 1, :]
        SV = V[:, i, :]

        w = A.dot(SV)  # sparse @ dense
        w -= beta[None, :] * SVold  # n x nv
        np.einsum('ij,ij->j', w, SV, out=alpha)

        T[:, i, i] = alpha

        if i < m - 1:
            w -= alpha[None, :] * SV  # n x nv
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
    elif normalization == 'none' or normalization is None:
        pass
    else:
        raise ValueError('Unknown normalization parameter!')
    return normed_msid


def _msid_descriptor(
        x: np.ndarray,
        ts: np.ndarray = np.logspace(-1, 1, 256),
        k: int = 5,
        m: int = 10,
        niters: int = 100,
        rademacher: bool = False,
        normalized_laplacian: bool = True,
        normalize: str = 'empty') -> np.ndarray:
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
        predicted_features: Low-dimension representation of predicted image set. Shape (N_pred, encoder_dim)
        target_features: Low-dimension representation of target image set. Shape (N_targ, encoder_dim)

    Returns:
        score: Scalar value of the distance between image sets features.

    Reference:
        https://arxiv.org/abs/1905.11141
    """

    def __init__(
            self,
            ts: torch.Tensor = torch.logspace(-1, 1, 256),
            k: int = 5,
            m: int = 10,
            niters: int = 100,
            rademacher: bool = False,
            normalized_laplacian: bool = True,
            normalize: str = 'empty',
            msid_mode: str = "max",
    ) -> None:
        r"""
        Args:
            ts: Temperature values.
            k: Number of neighbours for graph construction.
            m: Lanczos steps in SLQ.
            niters: Number of starting random vectors for SLQ.
            rademacher: True to use Rademacher distribution,
                False - standard normal for random vectors in Hutchinson.
            normalized_laplacian: if True, use normalized Laplacian.
            normalize: 'empty' for average heat kernel (corresponds to the empty graph normalization of NetLSD),
                    'complete' for the complete, 'er' for erdos-renyi normalization, 'none' for no normalization
            msid_mode: 'l2' to compute the l2 norm of the distance between `msid1` and `msid2`;
                    'max' to find the maximum abosulute difference between two descriptors over temperature
        """
        super(MSID, self).__init__()

        self.ts = ts.numpy()  # MSID works only with Numpy tensors
        self.k = k
        self.m = m
        self.niters = niters
        self.rademacher = rademacher
        self.msid_mode = msid_mode
        self.normalized_laplacian = normalized_laplacian
        self.normalize = normalize

    def compute_metric(self, predicted_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:

        r"""Compute MSID score between two sets of samples.
        Args:
            x: Samples from data distribution. Shape (N_samples, data_dim).
            y: Samples from data distribution. Shape (N_samples, data_dim).
            ts: Temperature values.
            k: Number of neighbours for graph construction.
            m: Lanczos steps in SLQ.
            niters: Number of starting random vectors for SLQ.
            rademacher: True to use Rademacher distribution,
                False - standard normal for random vectors in Hutchinson.
            msid_mode: 'l2' to compute the l2 norm of the distance between `msid1` and `msid2`;
                    'max' to find the maximum abosulute difference between two descriptors over temperature
            normalized_laplacian: if True, use normalized Laplacian.
            normalize: 'empty' for average heat kernel (corresponds to the empty graph normalization of NetLSD),
                    'complete' for the complete, 'er' for erdos-renyi normalization, 'none' for no normalization

        Returns:
            score: Scalar value of the distance between distributions.
        """
        normed_msid_pred = _msid_descriptor(
            predicted_features.detach().cpu().numpy(),
            ts=self.ts,
            k=self.k,
            m=self.m,
            niters=self.niters,
            rademacher=self.rademacher,
            normalized_laplacian=self.normalized_laplacian,
            normalize=self.normalize
        )
        normed_msid_target = _msid_descriptor(
            target_features.detach().cpu().numpy(),
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
            score = np.linalg.norm(normed_msid_pred - normed_msid_target)
        elif self.msid_mode == 'max':
            score = np.amax(c * np.abs(normed_msid_pred - normed_msid_target))
        else:
            raise ValueError('Mode must be in {`l2`, `max`}')

        return torch.tensor(score, device=predicted_features.device)
