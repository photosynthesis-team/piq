"""
PyTorch implementation of Frechet Inception Distance (FID score)
Reference:
    Martin Heusel et al. "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium",
    https://arxiv.org/abs/1706.08500
Credits:
    https://github.com/hukkelas/pytorch-frechet-inception-distance/
    https://github.com/mseitzer/pytorch-fid
"""

from typing import Tuple
import torch

from photosynthesis_metrics.base import BaseFeatureMetric


def _approximation_error(A: torch.Tensor, sA: torch.Tensor) -> torch.Tensor:
    normA = torch.norm(A)
    error = A - torch.mm(sA, sA)
    error = torch.norm(error) / normA
    return error


def _sqrtm_newton_schulz(A: torch.Tensor, num_iters: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Square root of matrix using Newton-Schulz Iterative method
    Source: https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py
    Args:
        A: matrix or batch of matrices
        num_iters: Number of iteration of the method
    Returns:
        Square root of matrix
        Error
    """
    expected_num_dims = 2
    if A.dim() != expected_num_dims:
        raise ValueError(f'Input dimension equals {A.dim()}, expected {expected_num_dims}')

    if num_iters <= 0:
        raise ValueError(f'Number of iteration equals {num_iters}, expected greater than 0')
    dtype = A.type()
    dim = A.size(0)
    normA = A.norm(p='fro')
    Y = A.div(normA)
    I = torch.eye(dim, dim, requires_grad=False).type(dtype)
    Z = torch.eye(dim, dim, requires_grad=False).type(dtype)

    sA = torch.empty_like(A)
    error = torch.empty(1)

    for i in range(num_iters):
        T = 0.5 * (3.0 * I - Z.mm(Y))
        Y = Y.mm(T)
        Z = T.mm(Z)

        sA = Y * torch.sqrt(normA)
        error = _approximation_error(A, sA)
        if torch.isclose(error, torch.tensor([0.], device=error.device), atol=1e-5):
            break
    return sA, error


def _compute_fid(mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor, sigma2: torch.Tensor,
                 eps=1e-6) -> torch.Tensor:
    r"""
    The Frechet Inception Distance between two multivariate Gaussians X_predicted ~ N(mu_1, sigm_1)
    and X_target ~ N(mu_2, sigm_2) is
        d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).

    Args:
        mu1: mean of activations calculated on predicted samples
        sigma1: covariance matrix over activations calculated on predicted samples
        mu2: mean of activations calculated on target samples
        sigma2: covariance matrix over activations calculated on target samples
        eps: offset constant. used if sigma_1 @ sigma_2 matrix is singular

    Returns:
        Scalar value of the distance between sets.
    """
    diff = mu1 - mu2
    covmean, _ = _sqrtm_newton_schulz(sigma1.mm(sigma2))
    # Product might be almost singular
    if not torch.isfinite(covmean).all():
        print(f'FID calculation produces singular product; adding {eps} to diagonal of cov estimates')
        offset = torch.eye(sigma1.size(0)) * eps
        covmean, _ = _sqrtm_newton_schulz((sigma1 + offset).mm(sigma2 + offset))

    tr_covmean = torch.trace(covmean)
    return diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean


def _cov(m: torch.Tensor, rowvar: bool = True) -> torch.Tensor:
    r"""Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    """
    if m.dim() > 2:
        raise ValueError('Tensor for covariance computations has more than 2 dimensions. '
                         'Only 1 or 2 dimensional arrays are allowed')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    return fact * m.matmul(mt).squeeze()


def _compute_statistics(samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Calculates the statistics used by FID
    Args:
        samples:  Low-dimension representation of image set.
            Shape (N_samples, dims) and dtype: np.float32 in range 0 - 1
    Returns:
        mu: mean over all activations from the encoder.
        sigma: covariance matrix over all activations from the encoder.
    """
    mu = torch.mean(samples, dim=0)
    sigma = _cov(samples, rowvar=False)
    return mu, sigma


class FID(BaseFeatureMetric):
    r"""
    Interface of Frechet Inception Distance.
    It's computed for a whole set of data and uses features from encoder instead of images itself to decrease
    computation cost. FID can compare two data distributions with different number of samples.
    But dimensionalities should match, otherwise it won't be possible to correctly compute statistics.

    Args:
        predicted_features: Low-dimension representation of predicted image set. Shape (N_pred, encoder_dim)
        target_features: Low-dimension representation of target image set. Shape (N_targ, encoder_dim)

    Returns:
        score: Scalar value of the distance between image sets features.


    References:
        .. [1] Heusel M. et al. (2017).
        Gans trained by a two time-scale update rule converge to a local nash equilibrium.
        Advances in neural information processing systems,
        https://arxiv.org/abs/1706.08500
    """

    def compute_metric(self, predicted_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        r"""
        Fits multivariate Gaussians: X ~ N(mu_1, sigm_1) and Y ~ N(mu_2, sigm_2) to image stacks.
        Then computes FID as d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).

        Args:
            predicted_features: Samples from data distribution.
                Shape (N_samples, data_dim), dtype: torch.float32 in range 0 - 1.
            target_features: Samples from data distribution.
                Shape (N_samples, data_dim), dtype: torch.float32 in range 0 - 1

        Returns:
        --   : The Frechet Distance.
        """
        # GPU -> CPU
        m_pred, s_pred = _compute_statistics(predicted_features.detach())
        m_targ, s_targ = _compute_statistics(target_features.detach())

        score = _compute_fid(m_pred, s_pred, m_targ, s_targ)

        return torch.tensor(score, device=predicted_features.device)
