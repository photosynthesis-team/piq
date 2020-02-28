r"""Implemetation of Multi-scale Evaluation metric, based on paper
 https://arxiv.org/abs/1905.11141 and author's repository https://github.com/xgfs/msid
"""
from functools import partial
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np

import scipy.sparse as sps
from scipy.sparse import lil_matrix, diags, eye
# from pykgraph import KGraph  # Not used for with sparce builder

from .utils import _validate_input
EPSILON = 1e-6
NORMALIZATION = 1e6

# ---- laplacian.py
def _np_euc_cdist(data : np.ndarray) -> np.ndarray: 
    dd = np.sum(data*data, axis=1)
    dist = -2*np.dot(data, data.T)
    dist += dd + dd[:, np.newaxis] 
    np.fill_diagonal(dist, 0)
    np.sqrt(dist, dist)
    return dist


def _construct_graph_sparse(data : np.ndarray, k: int):
    n = len(data)
    spmat = lil_matrix((n, n))
    dd = np.sum(data*data, axis=1)
    
    for i in range(n):
        dists = dd - 2*data[i, :].dot(data.T)
        inds = np.argpartition(dists, k+1)[:k+1]
        inds = inds[inds!=i]
        spmat[i, inds] = 1
            
    return spmat.tocsr()


# def _construct_graph_kgraph(data : np.ndarray, k : int):

#     n = len(data)
#     spmat = lil_matrix((n, n))
#     index = KGraph(data, 'euclidean')
#     index.build(reverse=0, K=2 * k + 1, L=2 * k + 50)
#     result = index.search(data, K=k + 1)[:, 1:]
#     spmat[np.repeat(np.arange(n), k, 0), result.ravel()] = 1
#     return spmat.tocsr()


def _laplacian_sparse(A : np.ndarray, normalized : bool = True):
    D = A.sum(1).A1
    if normalized:
        Dsqrt = diags(1/np.sqrt(D))
        L = eye(A.shape[0]) - Dsqrt.dot(A).dot(Dsqrt)
    else:
        L = diags(D) - A
    return L

## -------- slq.py

def _lanczos_m(A : np.ndarray, m : int, nv : int, rademacher : bool, SV : np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Lanczos algorithm computes symmetric m x m tridiagonal matrix T and matrix V with orthogonal rows
        constituting the basis of the Krylov subspace K_m(A, x),
        where x is an arbitrary starting unit vector.
        This implementation parallelizes `nv` starting vectors.
    
    Arguments:
        m: number of Lanczos steps
        nv: number of random vectors
        rademacher: True to use Rademacher distribution, 
                    False - standard normal for random vectors
        SV: specified starting vectors
    
    Returns:
        T: a nv x m x m tensor, T[i, :, :] is the ith symmetric tridiagonal matrix
        V: a n x m x nv tensor, V[:, :, i] is the ith matrix with orthogonal rows 
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


def _slq(A : np.ndarray, m : int, niters : int, rademacher : bool) -> np.ndarray:
    r"""
    Compute the trace of matrix exponential
    
    Arguments:
        A: square matrix in trace(exp(A))
        m: number of Lanczos steps
        niters: number of quadratures (also, the number of random vectors in the hutchinson trace estimator)
        rademacher: True to use Rademacher distribution, False - standard normal for random vectors in Hutchinson
    Returns:
        trace: estimate of trace of matrix exponential
    """
    T, _ = _lanczos_m(A, m, niters, rademacher)
    eigvals, eigvecs = np.linalg.eigh(T)
    expeig = np.exp(eigvals)
    sqeigv1 = np.power(eigvecs[:, 0, :], 2)
    trace = A.shape[-1] * (expeig * sqeigv1).sum() / niters
    return trace


def _slq_ts(A : np.ndarray, m : int, niters : int, ts : np.ndarray, rademacher : bool) -> np.ndarray:
    r"""
    Compute the trace of matrix exponential
    
    Arguments:
        A: square matrix in trace(exp(-t*A)), where t is temperature
        m: number of Lanczos steps
        niters: number of quadratures (also, the number of random vectors in the hutchinson trace estimator)
        ts: an array with temperatures
        rademacher: True to use Rademacher distribution, False - standard normal for random vectors in Hutchinson
    Returns:
        trace: estimate of trace of matrix exponential across temperatures `ts`
    """
    T, _ = _lanczos_m(A, m, niters, rademacher)
    eigvals, eigvecs = np.linalg.eigh(T)
    expeig = np.exp(-np.outer(ts, eigvals)).reshape(ts.shape[0], niters, m)
    sqeigv1 = np.power(eigvecs[:, 0, :], 2)
    traces = A.shape[-1] * (expeig * sqeigv1).sum(-1).mean(-1)
    return traces


def _slq_ts_fs(A : np.ndarray, m : int, niters : int, ts : np.ndarray, rademacher : bool, fs : List) -> np.ndarray:
    r"""
    Compute the trace of matrix functions
    
    Arguments:
        A: square matrix in trace(exp(-t*A)), where t is temperature
        m: number of Lanczos steps
        niters: number of quadratures (also, the number of random vectors in the hutchinson trace estimator)
        ts: an array with temperatures
        rademacher: True to use Rademacher distribution, else - standard normal for random vectors in Hutchinson
        fs: a list of functions
    Returns:
        traces: estimate of traces for each of the functions in fs
    """
    T, _ = _lanczos_m(A, m, niters, rademacher)
    eigvals, eigvecs = np.linalg.eigh(T)
    traces = np.zeros((len(fs), len(ts)))
    for i, f in enumerate(fs):
        expeig = f(-np.outer(ts, eigvals)).reshape(ts.shape[0], niters, m)
        sqeigv1 = np.power(eigvecs[:, 0, :], 2)
        traces[i, :] = A.shape[-1] * (expeig * sqeigv1).sum(-1).mean(-1)
    return traces


def slq_red_var(A : np.ndarray, m : int, niters : int, ts : np.ndarray, rademacher : bool) -> np.ndarray:
    r"""
    Compute the trace of matrix exponential with reduced variance
    
    Arguments:
        A: square matrix in trace(exp(-t*A)), where t is temperature
        m: number of Lanczos steps
        niters: number of quadratures (also, the number of random vectors in the hutchinson trace estimator)
        ts: an array with temperatures
    Returns:
        traces: estimate of trace for each temperature value in `ts`
    """
    fs = [np.exp, lambda x: x]

    traces = _slq_ts_fs(A, m, niters, ts, rademacher, fs)
    subee = traces[0, :] - traces[1, :] / np.exp(ts)
    sub = - ts * A.shape[0] / np.exp(ts)
    return subee + sub

# ---- msid.py

def _build_graph(data : np.ndarray, k : int = 5, graph_builder : str = 'sparse', normalized : bool = True):
    r"""
    Return Laplacian from data or load preconstructed from path
    Arguments:
        data: samples
        k: number of neighbours for graph construction
        graph_builder: if 'kgraph', use faster graph construction
        normalized: if True, use nnormalized Laplacian
    Returns:
        L: Laplacian of the graph constructed with data
    """
    if graph_builder == 'sparse':
        A = _construct_graph_sparse(data, k)
    # elif graph_builder == 'kgraph':
    #     A = _construct_graph_kgraph(data, k)
    else:
        raise ValueError('Specify graph builder: sparse or kgraph.')
    A = (A + A.T) / 2
    A.data = np.ones(A.data.shape)
    L = _laplacian_sparse(A, normalized)
    return L


def _normalize_msid(msid : np.ndarray, normalization : str, n : int, k : int, ts : np.ndarray):
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


def msid_score(
    x : np.ndarray, 
    y : np.ndarray, 
    ts : np.ndarray = np.logspace(-1, 1, 256),
    k : int = 5, m : int = 10, niters : int = 100, rademacher : bool = False, 
    graph_builder : str = 'sparse', msid_mode : str = 'max', normalized_laplacian : bool = True, normalize = 'empty'):

    r"""Compute the msid score between two samples, x and y
    Arguments:
        x: x samples
        y: y samples
        ts: temperature values
        k: number of neighbours for graph construction
        m: Lanczos steps in SLQ
        niters: number of starting random vectors for SLQ
        rademacher: if True, sample random vectors from Rademacher distributions, else sample standard normal distribution
        graph_builder: if 'kgraph', uses faster graph construction (options: 'sparse', 'kgraph')
        msid_mode: 'l2' to compute the l2 norm of the distance between `msid1` and `msid2`;
                'max' to find the maximum abosulute difference between two descriptors over temperature
        normalized_laplacian: if True, use normalized Laplacian
        normalize: 'empty' for average heat kernel (corresponds to the empty graph normalization of NetLSD),
                'complete' for the complete, 'er' for erdos-renyi
                normalization, 'none' for no normalization
    Returns:
        msid_score: the scalar value of the distance between discriptors
    """
    normed_msidx = msid_descriptor(x, ts, k, m, niters, rademacher, graph_builder, normalized_laplacian, normalize)
    normed_msidy = msid_descriptor(y, ts, k, m, niters, rademacher, graph_builder, normalized_laplacian, normalize)

    c = np.exp(-2 * (ts + 1 / ts))

    if msid_mode == 'l2':
        score = np.linalg.norm(normed_msidx - normed_msidy)
    elif msid_mode == 'max':
        score = np.amax(c * np.abs(normed_msidx - normed_msidy))
    else:
        raise ValueError('mode must be in {`l2`, `max`}')

    return score


def msid_descriptor(x : np.ndarray, ts : np.ndarray = np.logspace(-1, 1, 256), k : int = 5, m : int = 10, niters : int = 100, rademacher : bool =False, graph_builder : str = 'sparse',
              normalized_laplacian : bool = True, normalize : str = 'empty') -> np.ndarray:
    r"""
    Compute the msid descriptor for a single sample x
    Arguments:
        x: x samples
        ts: temperature values
        k: number of neighbours for graph construction
        m: Lanczos steps in SLQ
        niters: number of starting random vectors for SLQ
        rademacher: if True, sample random vectors from Rademacher distributions, else sample standard normal distribution
        graph_builder: if 'kgraph', uses faster graph construction (options: 'sparse', 'kgraph')
        normalized_laplacian: if True, use normalized Laplacian
        normalize: 'empty' for average heat kernel (corresponds to the empty graph normalization of NetLSD),
                'complete' for the complete, 'er' for erdos-renyi
                normalization, 'none' for no normalization
    Returns:
        normed_msidx: normalized msid descriptor
    """
    Lx = _build_graph(x, k, graph_builder, normalized_laplacian)

    nx = Lx.shape[0]
    msidx = slq_red_var(Lx, m, niters, ts, rademacher)

    normed_msidx = _normalize_msid(msidx, normalize, nx, k, ts) * NORMALIZATION

    return normed_msidx

class MSID(nn.Module):
    r"""Creates a criterion that measures MSID score for a batch of images
    See https://arxiv.org/abs/1905.11141 for reference.


    Args:
        ts: temperature values
        k: number of neighbours for graph construction
        m: Lanczos steps in SLQ
        niters: number of starting random vectors for SLQ
        rademacher: if True, sample random vectors from Rademacher distributions, else sample standard normal distribution
        graph_builder: if 'kgraph', uses faster graph construction (options: 'sparse', 'kgraph')
        normalized_laplacian: if True, use normalized Laplacian
        normalize: 'empty' for average heat kernel (corresponds to the empty graph normalization of NetLSD),
                'complete' for the complete, 'er' for erdos-renyi
                normalization, 'none' for no normalization
        msid_reduction: 'l2' to compute the l2 norm of the distance between `predictoon` and `target`;
                'max' to find the maximum abosulute difference between two descriptors over temperature
        channel_reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. 

    """

    def __init__(self, ts: List[float] = np.logspace(-1, 1, 256), k: int = 5, m: int = 10, niters: int = 100, rademacher: bool = False, graph_builder: str = 'sparse',
              normalized_laplacian: bool = True, normalize: str = 'empty', msid_reduction: str = "max", channel_reduction: str = 'mean'):

        super(MSID, self).__init__()

        self.msid_score = partial(msid_score, ts=ts, 
                                                    k=k, 
                                                    m=m, 
                                                    niters=niters, 
                                                    rademacher=rademacher, 
                                                    graph_builder=graph_builder, 
                                                    msid_mode=msid_reduction,
                                                    normalized_laplacian=normalized_laplacian, 
                                                    normalize=normalize)


        # Generic loss parameters.
        self.channel_reduction = channel_reduction

        # Loss-specific parameters.


    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        r"""Computation of MSID

        Args:
            x: Batch of images. Required to be 4D, channels first (N,C,H,W).
            y: Batch of images. Required to be 4D, channels first (N,C,H,W).

        Returns:
            normed_msidx: normalized msid descriptor (if no `target` specified).
            msid_score: the scalar value of the distance between discriptor if both `prediction` and `target` given.
        """

        N, _, _, _ = x.shape
        result = []
        for i in range(N):
            x_i = x[i]
            y_i = y[i]
            
            channels_x = [t.numpy() for t in map(torch.Tensor.squeeze_, torch.split(x_i, 1, dim=0))]
            channels_y = [t.numpy() for t in map(torch.Tensor.squeeze_, torch.split(y_i, 1, dim=0))]

            scores = torch.Tensor([self.msid_score(x, y) for x, y in zip(channels_x, channels_y)])
            
            if self.channel_reduction == 'mean':
                result.append(scores.mean())
            elif self.channel_reduction == 'sum':
                result.append(scores.sum())
            else:
                result.append(scores)

            return torch.Tensor(result)

def frechet_inception_distance(predicted_images: torch.Tensor, target_images: torch.Tensor,
                               feature_extractor: Optional[nn.Module] = None) -> float:
    assert isinstance(predicted_images, torch.Tensor) and isinstance(target_images, torch.Tensor), \
        f'Both image stacks must be of type torch.Tensor, got {type(predicted_images)} and {type(target_images)}.'

    # There is no assert for the full equality of shapes because it is not required.
    # Stacks can have any number of elements, but shapes of feature maps obtained from the images need to be equal.
    # Otherwise it will not be possible to correctly compute statistics.
    predicted_image_shape, target_image_shape = predicted_images.shape[1:], target_images.shape[1:]
    assert predicted_image_shape == target_image_shape, \
        f'Both image stacks must have images of the same shape, got {predicted_image_shape} and {target_image_shape}.'
    assert isinstance(feature_extractor, nn.Module) or feature_extractor is None, \
        f'Only PyTorch models are supported as feature extractors, got {type(feature_extractor)}.'

    if feature_extractor is None:
        print('WARNING: default feature extractor (InceptionNet V2) is used.')
        feature_extractor = InceptionV3()

    predicted_features = feature_extractor(predicted_images)
    target_features = feature_extractor(target_images)

    # TODO: `compute_fid` works with np.arrays, but need to work with torch.Tensors. Refactor that
    return compute_fid(predicted_features, target_features)