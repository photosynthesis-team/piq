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
import torch
from torch.nn.modules.loss import _Loss
from torch.utils.model_zoo import load_url
import torch.nn.functional as F
from photosynthesis_metrics.utils import _adjust_dimensions, _validate_input


def _ggd_parameters(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    gamma = torch.arange(0.2, 10 + 0.001, 0.001)
    r_table = (torch.lgamma(1. / gamma) + torch.lgamma(3. / gamma) - 2 * torch.lgamma(2. / gamma)).exp()
    r_table = r_table.repeat(x.size(0), 1)

    sigma_sq = x.pow(2).mean(dim=(-1, -2))
    sigma = sigma_sq.sqrt().squeeze(dim=-1)
    E = x.abs().mean(dim=(-1, -2))
    rho = sigma_sq / E ** 2

    indexes = (rho - r_table).abs().argmin(dim=-1)
    solution = gamma[indexes]

    return solution, sigma


def _aggd_parameters(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gamma = torch.arange(start=0.2, end=10.001, step=0.001)
    r_table = torch.exp(2 * torch.lgamma(2. / gamma) - torch.lgamma(1. / gamma) - torch.lgamma(3. / gamma)).repeat(
        x.size(0), 1)

    mask_left = x < 0
    mask_right = x > 0
    count_left = mask_left.sum(dim=(-1, -2))
    count_right = mask_right.sum(dim=(-1, -2))

    left_sigma = ((x * mask_left).pow(2).sum(dim=(-1, -2)) / count_left).sqrt()
    right_sigma = ((x * mask_right).pow(2).sum(dim=(-1, -2)) / count_right).sqrt()
    gamma_hat = left_sigma / right_sigma
    ro_hat = x.abs().mean(dim=(-1, -2)).pow(2) / x.pow(2).mean(dim=(-1, -2))
    ro_hat_norm = (ro_hat * (gamma_hat.pow(3) + 1) * (gamma_hat + 1)) / (gamma_hat.pow(2) + 1).pow(2)

    indexes = (ro_hat_norm - r_table).abs().argmin(dim=-1)
    solution = gamma[indexes]
    return solution, left_sigma.squeeze(dim=-1), right_sigma.squeeze(dim=-1)


def _gaussian_kernel2d(kernel_size: int = 7, sigma: float = 7 / 6) -> torch.Tensor:
    r"""Returns 2D Gaussian kernel N(0,`sigma`)
    Args:
        kernel_size: Size
        sigma: Sigma
    Returns:
        gaussian_kernel: 2D kernel with shape (kernel_size x kernel_size)

    """
    x = torch.arange(- (kernel_size // 2), kernel_size // 2 + 1).view(1, kernel_size)
    y = torch.arange(- (kernel_size // 2), kernel_size // 2 + 1).view(kernel_size, 1)
    kernel = torch.exp(-(x * x + y * y) / (2.0 * sigma ** 2))
    kernel = kernel / torch.sum(kernel)
    return kernel


def _natural_scene_statistics(luma: torch.Tensor, kernel_size: int = 7, sigma: float = 7. / 6) -> torch.Tensor:
    kernel = _gaussian_kernel2d(kernel_size=kernel_size, sigma=sigma).view(1, 1, kernel_size, kernel_size)
    C = 1
    mu = F.conv2d(luma, kernel, padding=kernel_size // 2)
    mu_sq = mu ** 2
    std = F.conv2d(luma ** 2, kernel, padding=kernel_size // 2)
    std = ((std - mu_sq).abs().sqrt())

    luma_nrmlzd = (luma - mu) / (std + C)
    features = []
    alpha, sigma = _ggd_parameters(luma_nrmlzd)
    features.extend((alpha, sigma.pow(2)))

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
                                   [2.8e-005, 0.530092], [0.001118, 0.370399]])

    scaled_features = lower_bound + (upper_bound - lower_bound) * (features - feature_ranges[..., 0]) / (
            feature_ranges[..., 1] - feature_ranges[..., 0])

    return scaled_features


def _RBF_kernel(features: torch.Tensor, sv: torch.Tensor, gamma: float = 0.05) -> torch.Tensor:
    features.unsqueeze_(dim=-1)
    sv.unsqueeze_(dim=0)
    dist = (features - sv).pow(2).sum(dim=1)
    return torch.exp(- dist * gamma)


def _score_svr(features: torch.Tensor) -> torch.Tensor:
    url = 'https://github.com/photosynthesis-team/photosynthesis.metrics/releases/' \
          'latest/download/brisque_svm_weights.pt'
    sv_coef, sv = load_url(url, map_location=features.device)

    # gamma and rho are SVM model parameters taken from official implementation of BRISQUE on MATLAB
    # Source: https://live.ece.utexas.edu/research/Quality/index_algorithms.htm
    gamma = 0.05
    rho = -153.591
    sv.t_()
    kernel_features = _RBF_kernel(features=features, sv=sv, gamma=gamma)
    score = kernel_features @ sv_coef
    return score - rho


def brisque(x: torch.Tensor,
            kernel_size: int = 7, kernel_sigma: float = 7 / 6,
            data_range: Union[int, float] = 1., reduction: str = 'mean',
            interpolation: str = 'nearest') -> torch.Tensor:
    r"""Interface of SBRISQUE index.
        Args:
            x: Batch of images. Required to be 2D (H, W), 3D (C,H,W) or 4D (N,C,H,W), channels first.
            kernel_size: The side-length of the sliding window used in comparison. Must be an odd value.
            kernel_sigma: Sigma of normal distribution.
            data_range: Value range of input images (usually 1.0 or 255).
            reduction: Reduction over samples in batch: "mean"|"sum"|"none".
            interpolation: Interpolation to be used for scaling.
        Returns:
            Value of BRISQUE index.
        References:
            .. [1] Anish Mittal et al. "No-Reference Image Quality Assessment in the Spatial Domain",
            https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf
        """
    _validate_input(input_tensors=x, allow_5d=False)
    x = _adjust_dimensions(input_tensors=x)

    x = x / data_range

    if x.size(1) == 3:
        # rgb_to_grey - weights to transform RGB image to grey
        rgb_to_grey = torch.tensor([0.299, 0.587, 0.114]).view(1, -1, 1, 1)
        x = torch.sum(x * rgb_to_grey, dim=1, keepdim=True)
    features = []
    num_of_scales = 2
    for iteration in range(num_of_scales):
        features.append(_natural_scene_statistics(x, kernel_size, kernel_sigma))
        x = F.interpolate(x, scale_factor=0.5, mode=interpolation)

    features = torch.cat(features, dim=-1)
    scaled_features = _scale_features(features)
    score = _score_svr(scaled_features)
    if reduction == 'none':
        return score

    return {'mean': score.mean,
            'sum': score.sum
            }[reduction](dim=0)


class BRISQUELoss(_Loss):
    r"""Creates a criterion that measures the BRISQUE score for input :math:`x`.

        :math:`x` is tensor of 2D (H, W), 3D (C,H,W) or 4D (N,C,H,W), channels first.

        The sum operation still operates over all the elements, and divides by :math:`n`.

        The division by :math:`n` can be avoided by setting ``reduction = 'sum'``.


        Args:
            kernel_size: By default, the mean and covariance of a pixel is obtained
                by convolution with given filter_size.
            kernel_sigma: Standard deviation for Gaussian kernel.
            data_range: The difference between the maximum and minimum of the pixel value,
                i.e., if for image x it holds min(x) = 0 and max(x) = 1, then data_range = 1.
                The pixel value interval of both input and output should remain the same.
            reduction: Specifies the reduction to apply to the output:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
                ``'mean'``: the sum of the output will be divided by the number of
                elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
                and :attr:`reduce` are in the process of being deprecated, and in the meantime,
                specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``.
            interpolation: Interpolation to be used for scaling.

        Shape:
            - Input: Required to be 2D (H, W), 3D (C,H,W) or 4D (N,C,H,W), channels first.

        Examples::

            >>> loss = BRISQUELoss()
            >>> prediction = torch.rand(3, 3, 256, 256, requires_grad=True)
            >>> target = torch.rand(3, 3, 256, 256)
            >>> output = loss(prediction)
            >>> output.backward()

        References:
            .. [1] Anish Mittal et al. "No-Reference Image Quality Assessment in the Spatial Domain",
            https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf
        """
    def __init__(self, kernel_size: int = 7, kernel_sigma: float = 7 / 6,
                 data_range: Union[int, float] = 1., reduction: str = 'mean',
                 interpolation: str = 'nearest') -> None:
        super().__init__()
        self.reduction = reduction
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self.data_range = data_range
        self.interpolation = interpolation

    def forward(self, prediction: torch.Tensor) -> torch.Tensor:
        r"""Computation of BRISQUE score as a loss function.

                Args:
                    prediction: Tensor of prediction of the network.

                Returns:
                    Value of BRISQUE loss to be minimized.
                """

        return brisque(prediction, reduction=self.reduction, kernel_size=self.kernel_size,
                       kernel_sigma=self.kernel_sigma, data_range=self.data_range, interpolation=self.interpolation)
