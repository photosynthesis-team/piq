r"""Implementation of Visual Information Fidelity metric
Code is based on MATLAB version for computations in pixel domain
https://live.ece.utexas.edu/research/Quality/VIF.htm

References:
    https://ieeexplore.ieee.org/abstract/document/1576816/
"""
import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from typing import Union

from piq.functional import gaussian_filter
from piq.utils import _validate_input, _reduce


def vif_p(x: torch.Tensor, y: torch.Tensor, sigma_n_sq: float = 2.0,
          data_range: Union[int, float] = 1.0, reduction: str = 'mean') -> torch.Tensor:
    r"""Compute Visiual Information Fidelity in **pixel** domain for a batch of images.
    This metric isn't symmetric, so make sure to place arguments in correct order.
    Both inputs supposed to have RGB channels order.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        sigma_n_sq: HVS model parameter (variance of the visual noise).
        data_range: Maximum value range of images (usually 1.0 or 255).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``

    Returns:
        VIF Index of similarity between two images. Usually in [0, 1] interval.
        Can be bigger than 1 for predicted :math:`x` images with higher contrast than original one.

    References:
        H. R. Sheikh and A. C. Bovik, "Image information and visual quality,"
        IEEE Transactions on Image Processing, vol. 15, no. 2, pp. 430-444, Feb. 2006
        https://ieeexplore.ieee.org/abstract/document/1576816/
        DOI: 10.1109/TIP.2005.859378.

    Note:
        In original paper this method was used for bands in discrete wavelet decomposition.
        Later on authors released code to compute VIF approximation in pixel domain.
        See https://live.ece.utexas.edu/research/Quality/VIF.htm for details.
    """
    _validate_input([x, y], dim_range=(4, 4), data_range=(0, data_range))

    min_size = 41
    if x.size(-1) < min_size or x.size(-2) < min_size:
        raise ValueError(f'Invalid size of the input images, expected at least {min_size}x{min_size}.')

    x = x / float(data_range) * 255
    y = y / float(data_range) * 255

    # Convert RGB image to YCbCr and take luminance: Y = 0.299 R + 0.587 G + 0.114 B
    num_channels = x.size(1)
    if num_channels == 3:
        x = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
        y = 0.299 * y[:, 0, :, :] + 0.587 * y[:, 1, :, :] + 0.114 * y[:, 2, :, :]

        # Add channel dimension
        x = x[:, None, :, :]
        y = y[:, None, :, :]

    # Constant for numerical stability
    EPS = 1e-8

    # Progressively downsample images and compute VIF on different scales
    x_vif, y_vif = 0, 0
    for scale in range(4):
        kernel_size = 2 ** (4 - scale) + 1
        kernel = gaussian_filter(kernel_size, sigma=kernel_size / 5)
        kernel = kernel.view(1, 1, kernel_size, kernel_size).to(x)

        if scale > 0:
            # Convolve and downsample
            x = F.conv2d(x, kernel)[:, :, ::2, ::2]  # valid padding
            y = F.conv2d(y, kernel)[:, :, ::2, ::2]  # valid padding

        mu_x, mu_y = F.conv2d(x, kernel), F.conv2d(y, kernel)  # valid padding
        mu_x_sq, mu_y_sq, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

        # Good
        sigma_x_sq = F.conv2d(x ** 2, kernel) - mu_x_sq
        sigma_y_sq = F.conv2d(y ** 2, kernel) - mu_y_sq
        sigma_xy = F.conv2d(x * y, kernel) - mu_xy

        # Zero small negative values
        sigma_x_sq = torch.relu(sigma_x_sq)
        sigma_y_sq = torch.relu(sigma_y_sq)

        g = sigma_xy / (sigma_y_sq + EPS)
        sigma_v_sq = sigma_x_sq - g * sigma_xy

        g = torch.where(sigma_y_sq >= EPS, g, torch.zeros_like(g))
        sigma_v_sq = torch.where(sigma_y_sq >= EPS, sigma_v_sq, sigma_x_sq)
        sigma_y_sq = torch.where(sigma_y_sq >= EPS, sigma_y_sq, torch.zeros_like(sigma_y_sq))

        g = torch.where(sigma_x_sq >= EPS, g, torch.zeros_like(g))
        sigma_v_sq = torch.where(sigma_x_sq >= EPS, sigma_v_sq, torch.zeros_like(sigma_v_sq))

        sigma_v_sq = torch.where(g >= 0, sigma_v_sq, sigma_x_sq)
        g = torch.relu(g)

        sigma_v_sq = torch.where(sigma_v_sq > EPS, sigma_v_sq, torch.ones_like(sigma_v_sq) * EPS)

        x_vif_scale = torch.log10(1.0 + (g ** 2.) * sigma_y_sq / (sigma_v_sq + sigma_n_sq))
        x_vif = x_vif + torch.sum(x_vif_scale, dim=[1, 2, 3])
        y_vif = y_vif + torch.sum(torch.log10(1.0 + sigma_y_sq / sigma_n_sq), dim=[1, 2, 3])

    score: torch.Tensor = (x_vif + EPS) / (y_vif + EPS)

    return _reduce(score, reduction)


class VIFLoss(_Loss):
    r"""Creates a criterion that measures the Visual Information Fidelity loss
    between predicted (x) and target (y) image. In order to be considered as a loss,
    value ``1 - clip(VIF, min=0, max=1)`` is returned.

    Args:
        sigma_n_sq: HVS model parameter (variance of the visual noise).
        data_range: Maximum value range of images (usually 1.0 or 255).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``

    Examples:
        >>> loss = VIFLoss()
        >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> y = torch.rand(3, 3, 256, 256)
        >>> output = loss(x, y)
        >>> output.backward()

    References:
        H. R. Sheikh and A. C. Bovik, "Image information and visual quality,"
        IEEE Transactions on Image Processing, vol. 15, no. 2, pp. 430-444, Feb. 2006
        https://ieeexplore.ieee.org/abstract/document/1576816/
        DOI: 10.1109/TIP.2005.859378.
    """

    def __init__(self, sigma_n_sq: float = 2.0, data_range: Union[int, float] = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.sigma_n_sq = sigma_n_sq
        self.data_range = data_range
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""Computation of Visual Information Fidelity (VIF) index as a loss function.
        Colour images are expected to have RGB channel order.
        Order of inputs is important! First tensor must contain distorted images, second reference images.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Value of VIF loss to be minimized in [0, 1] range.
        """
        # All checks are done in vif_p function
        score = vif_p(x, y, sigma_n_sq=self.sigma_n_sq, data_range=self.data_range, reduction=self.reduction)

        # Make sure value to be in [0, 1] range and convert to loss
        loss = 1 - torch.clamp(score, 0, 1)
        return loss
