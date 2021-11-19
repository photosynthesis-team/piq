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

from piq.base import BaseFeatureMetric
from piq.utils import _validate_input


def _approximation_error(matrix: torch.Tensor, s_matrix: torch.Tensor) -> torch.Tensor:
    norm_of_matrix = torch.norm(matrix)
    error = matrix - torch.mm(s_matrix, s_matrix)
    error = torch.norm(error) / norm_of_matrix
    return error


def _sqrtm_newton_schulz(matrix: torch.Tensor, num_iters: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Square root of matrix using Newton-Schulz Iterative method
    Source: https://github.com/msubhransu/matrix-sqrt/blob/master/matrix_sqrt.py
    Args:
        matrix: matrix or batch of matrices
        num_iters: Number of iteration of the method
    Returns:
        Square root of matrix
        Error
    """
    dim = matrix.size(0)
    norm_of_matrix = matrix.norm(p='fro')
    Y = matrix.div(norm_of_matrix)
    I = torch.eye(dim, dim, device=matrix.device, dtype=matrix.dtype)
    Z = torch.eye(dim, dim, device=matrix.device, dtype=matrix.dtype)

    s_matrix = torch.empty_like(matrix)
    error = torch.empty(1, device=matrix.device, dtype=matrix.dtype)

    for _ in range(num_iters):
        T = 0.5 * (3.0 * I - Z.mm(Y))
        Y = Y.mm(T)
        Z = T.mm(Z)

        s_matrix = Y * torch.sqrt(norm_of_matrix)
        error = _approximation_error(matrix, s_matrix)
        if torch.isclose(error, torch.tensor([0.], device=error.device, dtype=error.dtype), atol=1e-5):
            break

    return s_matrix, error


def _compute_fid(mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor, sigma2: torch.Tensor,
                 eps=1e-6) -> torch.Tensor:
    r"""
    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is
        d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).

    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples
        eps: offset constant. used if sigma_1 @ sigma_2 matrix is singular

    Returns:
        Scalar value of the distance between sets.
    """
    diff = mu1 - mu2
    covmean, _ = _sqrtm_newton_schulz(sigma1.mm(sigma2))

    # Product might be almost singular
    if not torch.isfinite(covmean).all():
        print(f'FID calculation produces singular product; adding {eps} to diagonal of cov estimates')
        offset = torch.eye(sigma1.size(0), device=mu1.device, dtype=mu1.dtype) * eps
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
    m = m - torch.mean(m, dim=1, keepdim=True)
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
    r"""Interface of Frechet Inception Distance.
    It's computed for a whole set of data and uses features from encoder instead of images itself to decrease
    computation cost. FID can compare two data distributions with different number of samples.
    But dimensionalities should match, otherwise it won't be possible to correctly compute statistics.

    Examples:
        >>> fid_metric = FID()
        >>> x_feats = torch.rand(10000, 1024)
        >>> y_feats = torch.rand(10000, 1024)
        >>> fid: torch.Tensor = fid_metric(x_feats, y_feats)

    References:
        Heusel M. et al. (2017).
        Gans trained by a two time-scale update rule converge to a local nash equilibrium.
        Advances in neural information processing systems,
        https://arxiv.org/abs/1706.08500
    """

    def compute_metric(self, x_features: torch.Tensor, y_features: torch.Tensor) -> torch.Tensor:
        r"""
        Fits multivariate Gaussians: :math:`X \sim \mathcal{N}(\mu_x, \sigma_x)` and
        :math:`Y \sim \mathcal{N}(\mu_y, \sigma_y)` to image stacks.
        Then computes FID as :math:`d^2 = ||\mu_x - \mu_y||^2 + Tr(\sigma_x + \sigma_y - 2\sqrt{\sigma_x \sigma_y})`.

        Args:
            x_features: Samples from data distribution. Shape :math:`(N_x, D)`
            y_features: Samples from data distribution. Shape :math:`(N_y, D)`

        Returns:
            The Frechet Distance.
        """
        _validate_input([x_features, y_features], dim_range=(2, 2), size_range=(1, 2))
        # GPU -> CPU
        mu_x, sigma_x = _compute_statistics(x_features.detach().to(dtype=torch.float64))
        mu_y, sigma_y = _compute_statistics(y_features.detach().to(dtype=torch.float64))

        score = _compute_fid(mu_x, sigma_x, mu_y, sigma_y)

        return score
