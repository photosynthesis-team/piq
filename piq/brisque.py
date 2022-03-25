"""
PyTorch implementation of BRISQUE
Reference:
    Anish Mittal et al. "No-Reference Image Quality Assessment in the Spatial Domain",
    https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf
Credits:
    https://live.ece.utexas.edu/research/Quality/index_algorithms.htm BRISQUE
    https://github.com/bukalapak/pybrisque
"""
from typing import Union, Tuple
import warnings
import torch
from torch.nn.modules.loss import _Loss
from torch.utils.model_zoo import load_url
import torch.nn.functional as F
from piq.utils import _validate_input, _reduce
from piq.functional import rgb2yiq, gaussian_filter


def brisque(x: torch.Tensor,
            kernel_size: int = 7, kernel_sigma: float = 7 / 6,
            data_range: Union[int, float] = 1., reduction: str = 'mean',
            interpolation: str = 'nearest') -> torch.Tensor:
    r"""Interface of BRISQUE index.
    Supports greyscale and colour images with RGB channel order.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        kernel_size: The side-length of the sliding window used in comparison. Must be an odd value.
        kernel_sigma: Sigma of normal distribution.
        data_range: Maximum value range of images (usually 1.0 or 255).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``
        interpolation: Interpolation to be used for scaling.

    Returns:
        Value of BRISQUE index.

    References:
        Anish Mittal et al. "No-Reference Image Quality Assessment in the Spatial Domain",
        https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf

    Note:
        The back propagation is not available using ``torch=1.5.0`` due to bug in ``argmin`` and ``argmax``
        backpropagation. Update the torch and torchvision to the latest versions.
    """
    if '1.5.0' in torch.__version__:
        warnings.warn(f'BRISQUE does not support back propagation due to bug in torch={torch.__version__}.'
                      f'Update torch to the latest version to access full functionality of the BRIQSUE.'
                      f'More info is available at https://github.com/photosynthesis-team/piq/pull/79 and'
                      f'https://github.com/pytorch/pytorch/issues/38869.')

    assert kernel_size % 2 == 1, f'Kernel size must be odd, got [{kernel_size}]'
    _validate_input([x, ], dim_range=(4, 4), data_range=(0, data_range))

    x = x / float(data_range) * 255

    if x.size(1) == 3:
        x = rgb2yiq(x)[:, :1]
    features = []
    num_of_scales = 2
    for _ in range(num_of_scales):
        features.append(_natural_scene_statistics(x, kernel_size, kernel_sigma))
        x = F.interpolate(x, size=(x.size(2) // 2, x.size(3) // 2), mode=interpolation)

    features = torch.cat(features, dim=-1)
    scaled_features = _scale_features(features)
    score = _score_svr(scaled_features)

    return _reduce(score, reduction)


class BRISQUELoss(_Loss):
    r"""Creates a criterion that measures the BRISQUE score for input :math:`x`.
    :math:`x` is 4D tensor (N, C, H, W).
    The sum operation still operates over all the elements, and divides by :math:`n`.
    The division by :math:`n` can be avoided by setting ``reduction = 'sum'``.

    Args:
        kernel_size: By default, the mean and covariance of a pixel is obtained
            by convolution with given filter_size. Must be an odd value.
        kernel_sigma: Standard deviation for Gaussian kernel.
        data_range: Maximum value range of images (usually 1.0 or 255).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``
        interpolation: Interpolation to be used for scaling.
    Examples:
        >>> loss = BRISQUELoss()
        >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> output = loss(x)
        >>> output.backward()
    References:
        Anish Mittal et al. "No-Reference Image Quality Assessment in the Spatial Domain",
        https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf

    Note:
        The back propagation is not available using ``torch=1.5.0`` due to bug in ``argmin`` and ``argmax``
        backpropagation. Update the torch and torchvision to the latest versions.
    """
    def __init__(self, kernel_size: int = 7, kernel_sigma: float = 7 / 6,
                 data_range: Union[int, float] = 1., reduction: str = 'mean',
                 interpolation: str = 'nearest') -> None:
        super().__init__()
        self.reduction = reduction
        self.kernel_size = kernel_size

        # This check might look redundant because kernel size is checked within the brisque function anyway.
        # However, this check allows to fail fast when the loss is being initialised and training has not been started.
        assert kernel_size % 2 == 1, f'Kernel size must be odd, got [{kernel_size}]'

        self.kernel_sigma = kernel_sigma
        self.data_range = data_range
        self.interpolation = interpolation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Computation of BRISQUE score as a loss function.

        Args:
            x: An input tensor with (N, C, H, W) shape. RGB channel order for colour images.

        Returns:
            Value of BRISQUE loss to be minimized.
        """
        return brisque(x, reduction=self.reduction, kernel_size=self.kernel_size,
                       kernel_sigma=self.kernel_sigma, data_range=self.data_range, interpolation=self.interpolation)


def _ggd_parameters(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    gamma = torch.arange(0.2, 10 + 0.001, 0.001).to(x)
    r_table = (torch.lgamma(1. / gamma) + torch.lgamma(3. / gamma) - 2 * torch.lgamma(2. / gamma)).exp()
    r_table = r_table.repeat(x.size(0), 1)

    sigma_sq = x.pow(2).mean(dim=(-1, -2))
    sigma = sigma_sq.sqrt().squeeze(dim=-1)

    assert not torch.isclose(sigma, torch.zeros_like(sigma)).all(), \
        'Expected image with non zero variance of pixel values'

    E = x.abs().mean(dim=(-1, -2))
    rho = sigma_sq / E ** 2

    indexes = (rho - r_table).abs().argmin(dim=-1)
    solution = gamma[indexes]
    return solution, sigma


def _aggd_parameters(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gamma = torch.arange(start=0.2, end=10.001, step=0.001).to(x)
    r_table = torch.exp(2 * torch.lgamma(2. / gamma) - torch.lgamma(1. / gamma) - torch.lgamma(3. / gamma))
    r_table = r_table.repeat(x.size(0), 1)

    mask_left = x < 0
    mask_right = x > 0
    count_left = mask_left.sum(dim=(-1, -2), dtype=torch.float32)
    count_right = mask_right.sum(dim=(-1, -2), dtype=torch.float32)

    assert (count_left > 0).all(), 'Expected input tensor (pairwise products of neighboring MSCN coefficients)' \
                                   '  with values below zero to compute parameters of AGGD'
    assert (count_right > 0).all(), 'Expected input tensor (pairwise products of neighboring MSCN coefficients)' \
                                    ' with values above zero to compute parameters of AGGD'

    left_sigma = ((x * mask_left).pow(2).sum(dim=(-1, -2)) / count_left).sqrt()
    right_sigma = ((x * mask_right).pow(2).sum(dim=(-1, -2)) / count_right).sqrt()

    assert (left_sigma > 0).all() and (right_sigma > 0).all(), f'Expected non-zero left and right variances, ' \
                                                               f'got {left_sigma} and {right_sigma}'

    gamma_hat = left_sigma / right_sigma
    ro_hat = x.abs().mean(dim=(-1, -2)).pow(2) / x.pow(2).mean(dim=(-1, -2))
    ro_hat_norm = (ro_hat * (gamma_hat.pow(3) + 1) * (gamma_hat + 1)) / (gamma_hat.pow(2) + 1).pow(2)

    indexes = (ro_hat_norm - r_table).abs().argmin(dim=-1)
    solution = gamma[indexes]
    return solution, left_sigma.squeeze(dim=-1), right_sigma.squeeze(dim=-1)


def _natural_scene_statistics(luma: torch.Tensor, kernel_size: int = 7, sigma: float = 7. / 6) -> torch.Tensor:
    kernel = gaussian_filter(kernel_size=kernel_size, sigma=sigma).view(1, 1, kernel_size, kernel_size).to(luma)
    C = 1
    mu = F.conv2d(luma, kernel, padding=kernel_size // 2)
    mu_sq = mu ** 2
    std = F.conv2d(luma ** 2, kernel, padding=kernel_size // 2)
    std = ((std - mu_sq).abs().sqrt())

    luma_nrmlzd = (luma - mu) / (std + C)
    alpha, sigma = _ggd_parameters(luma_nrmlzd)
    features = [alpha, sigma.pow(2)]

    shifts = [(0, 1), (1, 0), (1, 1), (-1, 1)]

    for shift in shifts:
        shifted_luma_nrmlzd = torch.roll(luma_nrmlzd, shifts=shift, dims=(-2, -1))
        alpha, sigma_l, sigma_r = _aggd_parameters(luma_nrmlzd * shifted_luma_nrmlzd)
        eta = (sigma_r - sigma_l) * torch.exp(
            torch.lgamma(2. / alpha) - (torch.lgamma(1. / alpha) + torch.lgamma(3. / alpha)) / 2)
        features.extend((alpha, eta, sigma_l.pow(2), sigma_r.pow(2)))

    return torch.stack(features, dim=-1)


def _scale_features(features: torch.Tensor) -> torch.Tensor:
    lower_bound = -1
    upper_bound = 1
    # Feature range is taken from official implementation of BRISQUE on MATLAB.
    # Source: https://live.ece.utexas.edu/research/Quality/index_algorithms.htm
    feature_ranges = torch.tensor([[0.338, 10], [0.017204, 0.806612], [0.236, 1.642],
                                   [-0.123884, 0.20293], [0.000155, 0.712298], [0.001122, 0.470257],
                                   [0.244, 1.641], [-0.123586, 0.179083], [0.000152, 0.710456],
                                   [0.000975, 0.470984], [0.249, 1.555], [-0.135687, 0.100858],
                                   [0.000174, 0.684173], [0.000913, 0.534174], [0.258, 1.561],
                                   [-0.143408, 0.100486], [0.000179, 0.685696], [0.000888, 0.536508],
                                   [0.471, 3.264], [0.012809, 0.703171], [0.218, 1.046],
                                   [-0.094876, 0.187459], [1.5e-005, 0.442057], [0.001272, 0.40803],
                                   [0.222, 1.042], [-0.115772, 0.162604], [1.6e-005, 0.444362],
                                   [0.001374, 0.40243], [0.227, 0.996],
                                   [-0.117188, 0.09832299999999999], [3e-005, 0.531903],
                                   [0.001122, 0.369589], [0.228, 0.99], [-0.12243, 0.098658],
                                   [2.8e-005, 0.530092], [0.001118, 0.370399]]).to(features)

    scaled_features = lower_bound + (upper_bound - lower_bound) * (features - feature_ranges[..., 0]) / (
            feature_ranges[..., 1] - feature_ranges[..., 0])

    return scaled_features


def _rbf_kernel(features: torch.Tensor, sv: torch.Tensor, gamma: float = 0.05) -> torch.Tensor:
    dist = (features.unsqueeze(dim=-1) - sv.unsqueeze(dim=0)).pow(2).sum(dim=1)
    return torch.exp(- dist * gamma)


def _score_svr(features: torch.Tensor) -> torch.Tensor:
    url = 'https://github.com/photosynthesis-team/piq/' \
          'releases/download/v0.4.0/brisque_svm_weights.pt'
    sv_coef, sv = load_url(url, map_location=features.device)

    # gamma and rho are SVM model parameters taken from official implementation of BRISQUE on MATLAB
    # Source: https://live.ece.utexas.edu/research/Quality/index_algorithms.htm
    gamma = 0.05
    rho = -153.591
    sv.t_()
    kernel_features = _rbf_kernel(features=features, sv=sv, gamma=gamma)
    score = kernel_features @ sv_coef.to(dtype=features.dtype)
    return score - rho
