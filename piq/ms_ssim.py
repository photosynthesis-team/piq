r""" This module implements Multi-Scale Structural Similarity (MS-SSIM) index in PyTorch.

Implementation of classes and functions from this module are inspired by Gongfan Fang's (@VainF) implementation:
https://github.com/VainF/pytorch-msssim

and implementation of one of pull requests to the PyTorch by Kangfu Mei (@MKFMIKU):
https://github.com/pytorch/pytorch/pull/22289/files
"""
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from piq.utils import _validate_input, _reduce
from piq.functional import gaussian_filter
from piq.ssim import _ssim_per_channel, _ssim_per_channel_complex


def multi_scale_ssim(x: torch.Tensor, y: torch.Tensor, kernel_size: int = 11, kernel_sigma: float = 1.5,
                     data_range: Union[int, float] = 1., reduction: str = 'mean',
                     scale_weights: Optional[torch.Tensor] = None,
                     k1: float = 0.01, k2: float = 0.03) -> torch.Tensor:
    r""" Interface of Multi-scale Structural Similarity (MS-SSIM) index.
    Inputs supposed to be in range ``[0, data_range]`` with RGB channels order for colour images.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)` or :math:`(N, C, H, W, 2)`.
        y: A target tensor. Shape :math:`(N, C, H, W)` or :math:`(N, C, H, W, 2)`.
        kernel_size: The side-length of the sliding window used in comparison. Must be an odd value.
        kernel_sigma: Sigma of normal distribution.
        data_range: Maximum value range of images (usually 1.0 or 255).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        scale_weights: Weights for different scales.
            If ``None``, default weights from the paper will be used.
            Default weights: (0.0448, 0.2856, 0.3001, 0.2363, 0.1333).
        k1: Algorithm parameter, K1 (small constant).
        k2: Algorithm parameter, K2 (small constant).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Value of Multi-scale Structural Similarity (MS-SSIM) index. In case of 5D input tensors,
        complex value is returned as a tensor of size 2.

    References:
        Wang, Z., Simoncelli, E. P., Bovik, A. C. (2003).
        Multi-scale Structural Similarity for Image Quality Assessment.
        IEEE Asilomar Conference on Signals, Systems and Computers, 37,
        https://ieeexplore.ieee.org/document/1292216 DOI:`10.1109/ACSSC.2003.1292216`

        Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
        Image quality assessment: From error visibility to structural similarity.
        IEEE Transactions on Image Processing, 13, 600-612.
        https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
        DOI: `10.1109/TIP.2003.819861`
    Note:
        The size of the image should be at least ``(kernel_size - 1) * 2 ** (levels - 1) + 1``.
    """
    assert kernel_size % 2 == 1, f'Kernel size must be odd, got [{kernel_size}]'
    _validate_input([x, y], dim_range=(4, 5), data_range=(0, data_range))

    x = x / float(data_range)
    y = y / float(data_range)

    if scale_weights is None:
        # Values from MS-SSIM the paper
        scale_weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(x)
    else:
        # Normalize scale weights
        scale_weights = (scale_weights / scale_weights.sum()).to(x)

    kernel = gaussian_filter(kernel_size, kernel_sigma).repeat(x.size(1), 1, 1, 1).to(x)

    _compute_msssim = _multi_scale_ssim_complex if x.dim() == 5 else _multi_scale_ssim
    msssim_val = _compute_msssim(
        x=x,
        y=y,
        data_range=data_range,
        kernel=kernel,
        scale_weights=scale_weights,
        k1=k1,
        k2=k2
    )
    return _reduce(msssim_val, reduction)


class MultiScaleSSIMLoss(_Loss):
    r"""Creates a criterion that measures the multi-scale structural similarity index error between
    each element in the input :math:`x` and target :math:`y`.
    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        MSSIM = \{mssim_1,\dots,mssim_{N \times C}\}, \\
        mssim_{l}(x, y) = \frac{(2 \mu_{x,m} \mu_{y,m} + c_1) }
        {(\mu_{x,m}^2 +\mu_{y,m}^2 + c_1)} \prod_{j=1}^{m - 1}
        \frac{(2 \sigma_{xy,j} + c_2)}{(\sigma_{x,j}^2 +\sigma_{y,j}^2 + c_2)}

    where :math:`N` is the batch size, `C` is the channel size, `m` is the scale level (Default: 5).
    If :attr:`reduction` is not ``'none'`` (default ``'mean'``), then:

    .. math::
        MultiscaleSSIMLoss(x, y) =
        \begin{cases}
            \operatorname{mean}(1 - MSSIM), &  \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(1 - MSSIM),  &  \text{if reduction} = \text{'sum'.}
        \end{cases}

    For colour images channel order is RGB.
    In case of 5D input tensors, complex value is returned as a tensor of size 2.

    Args:
        kernel_size: By default, the mean and covariance of a pixel is obtained
            by convolution with given filter_size. Must be an odd value.
        kernel_sigma: Standard deviation for Gaussian kernel.
        k1: Coefficient related to c1 in the above equation.
        k2: Coefficient related to c2 in the above equation.
        scale_weights:  Weights for different scales.
            If ``None``, default weights from the paper will be used.
            Default weights: (0.0448, 0.2856, 0.3001, 0.2363, 0.1333).
        reduction: Specifies the reduction type: ``'none'`` | ``'mean'`` | ``'sum'``.
            Default: ``'mean'``
        data_range: Maximum value range of images (usually 1.0 or 255).

    Examples:
        >>> loss = MultiScaleSSIMLoss()
        >>> input = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> target = torch.rand(3, 3, 256, 256)
        >>> output = loss(input, target)
        >>> output.backward()

    References:
        Wang, Z., Simoncelli, E. P., Bovik, A. C. (2003).
        Multi-scale Structural Similarity for Image Quality Assessment.
        IEEE Asilomar Conference on Signals, Systems and Computers, 37,
        https://ieeexplore.ieee.org/document/1292216
        DOI:`10.1109/ACSSC.2003.1292216`

        Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
        Image quality assessment: From error visibility to structural similarity.
        IEEE Transactions on Image Processing, 13, 600-612.
        https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
        DOI:`10.1109/TIP.2003.819861`

    Note:
        The size of the image should be at least ``(kernel_size - 1) * 2 ** (levels - 1) + 1``.
    """
    __constants__ = ['kernel_size', 'k1', 'k2', 'sigma', 'kernel', 'reduction']

    def __init__(self, kernel_size: int = 11, kernel_sigma: float = 1.5, k1: float = 0.01, k2: float = 0.03,
                 scale_weights: Optional[torch.Tensor] = None,
                 reduction: str = 'mean', data_range: Union[int, float] = 1.) -> None:
        super().__init__()

        # Generic loss parameters.
        self.reduction = reduction

        # Loss-specific parameters.
        if scale_weights is None:
            # Values from MS-SSIM paper
            self.scale_weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
        else:
            self.scale_weights = scale_weights

        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma

        # This check might look redundant because kernel size is checked within the ms-ssim function anyway.
        # However, this check allows to fail fast when the loss is being initialised and training has not been started.
        assert kernel_size % 2 == 1, f'Kernel size must be odd, got [{kernel_size}]'

        self.k1 = k1
        self.k2 = k2
        self.data_range = data_range

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""Computation of Multi-scale Structural Similarity (MS-SSIM) index as a loss function.
        For colour images channel order is RGB.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)` or :math:`(N, C, H, W, 2)`.
            y: A target tensor. Shape :math:`(N, C, H, W)` or :math:`(N, C, H, W, 2)`.

        Returns:
            Value of MS-SSIM loss to be minimized, i.e. ``1 - ms_ssim`` in [0, 1] range. In case of 5D tensor,
            complex value is returned as a tensor of size 2.
        """
        score = multi_scale_ssim(x=x, y=y, kernel_size=self.kernel_size, kernel_sigma=self.kernel_sigma,
                                 data_range=self.data_range, reduction=self.reduction, scale_weights=self.scale_weights,
                                 k1=self.k1, k2=self.k2)
        return torch.ones_like(score) - score


def _multi_scale_ssim(x: torch.Tensor, y: torch.Tensor, data_range: Union[int, float], kernel: torch.Tensor,
                      scale_weights: torch.Tensor, k1: float, k2: float) -> torch.Tensor:
    r"""Calculates Multi scale Structural Similarity (MS-SSIM) index for X and Y.
    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        data_range: Maximum value range of images (usually 1.0 or 255).
        kernel: 2D Gaussian kernel.
        scale_weights: Weights for scaled SSIM
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        Value of Multi scale Structural Similarity (MS-SSIM) index.
    """
    levels = scale_weights.size(0)
    min_size = (kernel.size(-1) - 1) * 2 ** (levels - 1) + 1
    if x.size(-1) < min_size or x.size(-2) < min_size:
        raise ValueError(f'Invalid size of the input images, expected at least {min_size}x{min_size}.')

    mcs = []
    ssim_val = None
    for iteration in range(levels):
        if iteration > 0:
            padding = max(x.shape[2] % 2, x.shape[3] % 2)
            x = F.pad(x, pad=[padding, 0, padding, 0], mode='replicate')
            y = F.pad(y, pad=[padding, 0, padding, 0], mode='replicate')
            x = F.avg_pool2d(x, kernel_size=2, padding=0)
            y = F.avg_pool2d(y, kernel_size=2, padding=0)

        ssim_val, cs = _ssim_per_channel(x, y, kernel=kernel, data_range=data_range, k1=k1, k2=k2)
        mcs.append(cs)

    # mcs, (level, batch)
    mcs_ssim = torch.relu(torch.stack(mcs[:-1] + [ssim_val], dim=0))

    # weights, (level)
    msssim_val = torch.prod((mcs_ssim ** scale_weights.view(-1, 1, 1)), dim=0).mean(1)

    return msssim_val


def _multi_scale_ssim_complex(x: torch.Tensor, y: torch.Tensor, data_range: Union[int, float],
                              kernel: torch.Tensor, scale_weights: torch.Tensor, k1: float,
                              k2: float) -> torch.Tensor:
    r"""Calculate Multi scale Structural Similarity (MS-SSIM) index for Complex X and Y.
    Args:
        x: An input tensor. Shape :math:`(N, C, H, W, 2)`.
        y: A target tensor. Shape :math:`(N, C, H, W, 2)`.
        data_range: Maximum value range of images (usually 1.0 or 255).
        kernel: 2-D gauss kernel.
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        Value of Complex Multi scale Structural Similarity (MS-SSIM) index.
    """
    levels = scale_weights.size(0)
    min_size = (kernel.size(-1) - 1) * 2 ** (levels - 1) + 1
    if x.size(-2) < min_size or x.size(-3) < min_size:
        raise ValueError(f'Invalid size of the input images, expected at least {min_size}x{min_size}.')
    mcs = []
    ssim_val = None
    for iteration in range(levels):
        x_real = x[..., 0]
        x_imag = x[..., 1]
        y_real = y[..., 0]
        y_imag = y[..., 1]
        if iteration > 0:
            padding = max(x.size(2) % 2, x.size(3) % 2)
            x_real = F.pad(x_real, pad=[padding, 0, padding, 0], mode='replicate')
            x_imag = F.pad(x_imag, pad=[padding, 0, padding, 0], mode='replicate')
            y_real = F.pad(y_real, pad=[padding, 0, padding, 0], mode='replicate')
            y_imag = F.pad(y_imag, pad=[padding, 0, padding, 0], mode='replicate')

            x_real = F.avg_pool2d(x_real, kernel_size=2, padding=0)
            x_imag = F.avg_pool2d(x_imag, kernel_size=2, padding=0)
            y_real = F.avg_pool2d(y_real, kernel_size=2, padding=0)
            y_imag = F.avg_pool2d(y_imag, kernel_size=2, padding=0)
            x = torch.stack((x_real, x_imag), dim=-1)
            y = torch.stack((y_real, y_imag), dim=-1)

        ssim_val, cs = _ssim_per_channel_complex(x, y, kernel=kernel, data_range=data_range, k1=k1, k2=k2)
        mcs.append(cs)

    # mcs, (level, batch)
    mcs_ssim = torch.relu(torch.stack(mcs[:-1] + [ssim_val], dim=0))

    mcs_ssim_real = mcs_ssim[..., 0]
    mcs_ssim_imag = mcs_ssim[..., 1]
    mcs_ssim_abs = (mcs_ssim_real.pow(2) + mcs_ssim_imag.pow(2)).sqrt()
    mcs_ssim_deg = torch.atan2(mcs_ssim_imag, mcs_ssim_real)

    mcs_ssim_pow_abs = mcs_ssim_abs ** scale_weights.view(-1, 1, 1)
    mcs_ssim_pow_deg = mcs_ssim_deg * scale_weights.view(-1, 1, 1)

    msssim_val_abs = torch.prod(mcs_ssim_pow_abs, dim=0)
    msssim_val_deg = torch.sum(mcs_ssim_pow_deg, dim=0)
    msssim_val_real = msssim_val_abs * torch.cos(msssim_val_deg)
    msssim_val_imag = msssim_val_abs * torch.sin(msssim_val_deg)
    msssim_val = torch.stack((msssim_val_real, msssim_val_imag), dim=-1).mean(dim=1)
    return msssim_val
