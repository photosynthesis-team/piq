r""" This module implements Structural Similarity (SSIM) index in PyTorch.

Implementation of classes and functions from this module are inspired by Gongfan Fang's (@VainF) implementation:
https://github.com/VainF/pytorch-msssim

and implementation of one of pull requests to the PyTorch by Kangfu Mei (@MKFMIKU):
https://github.com/pytorch/pytorch/pull/22289/files
"""
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from piq.utils import _adjust_dimensions, _validate_input
from piq.functional import gaussian_filter


def ssim(x: torch.Tensor, y: torch.Tensor, kernel_size: int = 11, kernel_sigma: float = 1.5,
         data_range: Union[int, float] = 1., reduction: str = 'mean', full: bool = False,
         k1: float = 0.01, k2: float = 0.03) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Interface of Structural Similarity (SSIM) index.

    Args:
        x: Batch of images. Required to be 2D (H, W), 3D (C,H,W), 4D (N,C,H,W) or 5D (N,C,H,W,2), channels first.
        y: Batch of images. Required to be 2D (H, W), 3D (C,H,W) 4D (N,C,H,W) or 5D (N,C,H,W,2), channels first.
        kernel_size: The side-length of the sliding window used in comparison. Must be an odd value.
        kernel_sigma: Sigma of normal distribution.
        data_range: Value range of input images (usually 1.0 or 255).
        reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        full: Return cs map or not.
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Value of Structural Similarity (SSIM) index. In case of 5D input tensors, complex value is returned
        as a tensor of size 2.

    References:
        .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
           (2004). Image quality assessment: From error visibility to
           structural similarity. IEEE Transactions on Image Processing,
           13, 600-612.
           https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
           :DOI:`10.1109/TIP.2003.819861`
    """
    _validate_input(input_tensors=(x, y), allow_5d=True, kernel_size=kernel_size, scale_weights=None)
    x, y = _adjust_dimensions(input_tensors=(x, y))
    if isinstance(x, torch.ByteTensor) or isinstance(y, torch.ByteTensor):
        x = x.type(torch.float32)
        y = y.type(torch.float32)

    kernel = gaussian_filter(kernel_size, kernel_sigma).repeat(x.size(1), 1, 1, 1).to(y)
    _compute_ssim_per_channel = _ssim_per_channel_complex if x.dim() == 5 else _ssim_per_channel
    ssim_map, cs_map = _compute_ssim_per_channel(x=x, y=y, kernel=kernel, data_range=data_range, k1=k1, k2=k2)
    ssim_val = ssim_map.mean(1)
    cs = cs_map.mean(1)

    if reduction != 'none':
        reduction_operation = {'mean': torch.mean,
                               'sum': torch.sum}
        ssim_val = reduction_operation[reduction](ssim_val, dim=0)
        cs = reduction_operation[reduction](cs, dim=0)

    if full:
        return ssim_val, cs

    return ssim_val


class SSIMLoss(_Loss):
    r"""Creates a criterion that measures the structural similarity index error between
    each element in the input :math:`x` and target :math:`y`.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        SSIM = \{ssim_1,\dots,ssim_{N \times C}\}, \quad
        ssim_{l}(x, y) = \frac{(2 \mu_x \mu_y + c_1) (2 \sigma_{xy} + c_2)}
        {(\mu_x^2 +\mu_y^2 + c_1)(\sigma_x^2 +\sigma_y^2 + c_2)},

    where :math:`N` is the batch size, `C` is the channel size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        SSIMLoss(x, y) =
        \begin{cases}
            \operatorname{mean}(1 - SSIM), &  \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(1 - SSIM),  &  \text{if reduction} = \text{'sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    The sum operation still operates over all the elements, and divides by :math:`n`.
    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.
    In case of 5D input tensors, complex value is returned as a tensor of size 2.

    Args:
        kernel_size: By default, the mean and covariance of a pixel is obtained
            by convolution with given filter_size.
        kernel_sigma: Standard deviation for Gaussian kernel.
        k1: Coefficient related to c1 in the above equation.
        k2: Coefficient related to c2 in the above equation.
        reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        data_range: The difference between the maximum and minimum of the pixel value,
            i.e., if for image x it holds min(x) = 0 and max(x) = 1, then data_range = 1.
            The pixel value interval of both input and output should remain the same.

    Shape:
        - Input: Required to be 2D (H, W), 3D (C,H,W), 4D (N,C,H,W) or 5D (N,C,H,W,2), channels first.
        - Target: Required to be 2D (H, W), 3D (C,H,W), 4D (N,C,H,W) or 5D (N,C,H,W,2), channels first.

    Examples::
        >>> loss = SSIMLoss()
        >>> prediction = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> target = torch.rand(3, 3, 256, 256)
        >>> output = loss(prediction, target)
        >>> output.backward()

    References:
        .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
           (2004). Image quality assessment: From error visibility to
           structural similarity. IEEE Transactions on Image Processing,
           13, 600-612.
           https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
           :DOI:`10.1109/TIP.2003.819861`
    """
    __constants__ = ['kernel_size', 'k1', 'k2', 'sigma', 'kernel', 'reduction']

    def __init__(self, kernel_size: int = 11, kernel_sigma: float = 1.5, k1: float = 0.01, k2: float = 0.03,
                 reduction: str = 'mean', data_range: Union[int, float] = 1.) -> None:
        super().__init__()

        # Generic loss parameters.
        self.reduction = reduction

        # Loss-specific parameters.
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self.k1 = k1
        self.k2 = k2
        self.data_range = data_range

    def forward(self,
                prediction: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        r"""Computation of Structural Similarity (SSIM) index as a loss function.

        Args:
            prediction: Tensor of prediction of the network. Required to be
                2D (H, W), 3D (C,H,W), 4D (N,C,H,W) or 5D (N,C,H,W,2), channels first.
            target: Reference tensor. Required to be 2D (H, W), 3D (C,H,W), 4D (N,C,H,W) or 5D (N,C,H,W,2),
                channels first.

        Returns:
            Value of SSIM loss to be minimized, i.e 1 - `ssim`. 0 <= SSIM loss <= 1. In case of 5D input tensors,
            complex value is returned as a tensor of size 2.
        """

        score = ssim(x=prediction, y=target, kernel_size=self.kernel_size, kernel_sigma=self.kernel_sigma,
                     data_range=self.data_range, reduction=self.reduction, full=False, k1=self.k1, k2=self.k2)
        return torch.ones_like(score) - score


def multi_scale_ssim(x: torch.Tensor, y: torch.Tensor, kernel_size: int = 11, kernel_sigma: float = 1.5,
                     data_range: Union[int, float] = 1., reduction: str = 'mean',
                     scale_weights: Optional[Union[Tuple[float], List[float], torch.Tensor]] = None,
                     k1: float = 0.01, k2: float = 0.03) -> torch.Tensor:
    r""" Interface of Multi-scale Structural Similarity (MS-SSIM) index.

    Args:
        x: Batch of images. Required to be 2D (H, W), 3D (C,H,W), 4D (N,C,H,W) or 5D (N,C,H,W,2), channels first.
            The size of the image should be (kernel_size - 1) * 2 ** (levels - 1) + 1.
        y: Batch of images. Required to be 2D (H, W), 3D (C,H,W), 4D (N,C,H,W) or 5D (N,C,H,W,2), channels first.
            The size of the image should be (kernel_size - 1) * 2 ** (levels - 1) + 1.
        kernel_size: The side-length of the sliding window used in comparison. Must be an odd value.
        kernel_sigma: Sigma of normal distribution.
        data_range: Value range of input images (usually 1.0 or 255).
        reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        scale_weights: Weights for different scales.
            If None, default weights from the paper [1] will be used.
            Default weights: (0.0448, 0.2856, 0.3001, 0.2363, 0.1333).
        k1: Algorithm parameter, K1 (small constant, see [2]).
        k2: Algorithm parameter, K2 (small constant, see [2]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Value of Multi-scale Structural Similarity (MS-SSIM) index. In case of 5D input tensors,
        complex value is returned as a tensor of size 2.

    References:
        .. [1] Wang, Z., Simoncelli, E. P., Bovik, A. C. (2003).
           Multi-scale Structural Similarity for Image Quality Assessment.
           IEEE Asilomar Conference on Signals, Systems and Computers, 37,
           https://ieeexplore.ieee.org/document/1292216
           :DOI:`10.1109/ACSSC.2003.1292216`
        .. [2] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
           (2004). Image quality assessment: From error visibility to
           structural similarity. IEEE Transactions on Image Processing,
           13, 600-612.
           https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
           :DOI:`10.1109/TIP.2003.819861`
    """
    _validate_input(input_tensors=(x, y), allow_5d=True, kernel_size=kernel_size, scale_weights=scale_weights)
    x, y = _adjust_dimensions(input_tensors=(x, y))
    if isinstance(x, torch.ByteTensor) or isinstance(y, torch.ByteTensor):
        x = x.type(torch.float32)
        y = y.type(torch.float32)

    if scale_weights is None:
        scale_weights_from_ms_ssim_paper = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        scale_weights = scale_weights_from_ms_ssim_paper

    scale_weights_tensor = scale_weights if isinstance(scale_weights, torch.Tensor) else torch.tensor(scale_weights)
    scale_weights_tensor = scale_weights_tensor.to(y)
    kernel = gaussian_filter(kernel_size, kernel_sigma).repeat(x.size(1), 1, 1, 1).to(y)
    
    _compute_msssim = _multi_scale_ssim_complex if x.dim() == 5 else _multi_scale_ssim
    msssim_val = _compute_msssim(
        x=x,
        y=y,
        data_range=data_range,
        kernel=kernel,
        scale_weights_tensor=scale_weights_tensor,
        k1=k1,
        k2=k2
    )

    if reduction == 'none':
        return msssim_val

    return {'mean': torch.mean,
            'sum': torch.sum}[reduction](msssim_val, dim=0)


class MultiScaleSSIMLoss(_Loss):
    r"""Creates a criterion that measures the multi-scale structural similarity index error between
    each element in the input :math:`x` and target :math:`y`.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        MSSIM = \{mssim_1,\dots,mssim_{N \times C}\}, \quad
        mssim_{l}(x, y) = \frac{(2 \mu_{x,m} \mu_{y,m} + c_1) }
        {(\mu_{x,m}^2 +\mu_{y,m}^2 + c_1)} \prod_{j=1}^{m - 1}
        \frac{(2 \sigma_{xy,j} + c_2)}{(\sigma_{x,j}^2 +\sigma_{y,j}^2 + c_2)}

    where :math:`N` is the batch size, `C` is the channel size, `m` is the scale level (Default: 5).
    If :attr:`reduction` is not ``'none'``(default ``'mean'``), then:

    .. math::
        MultiscaleSSIMLoss(x, y) =
        \begin{cases}
            \operatorname{mean}(1 - MSSIM), &  \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(1 - MSSIM),  &  \text{if reduction} = \text{'sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.
    The sum operation still operates over all the elements, and divides by :math:`n`.
    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.
    In case of 5D input tensors, complex value is returned as a tensor of size 2.

    Args:
        kernel_size: By default, the mean and covariance of a pixel is obtained
            by convolution with given filter_size.
        kernel_sigma: Standard deviation for Gaussian kernel.
        k1: Coefficient related to c1 in the above equation.
        k2: Coefficient related to c2 in the above equation.
        scale_weights:  Weights for different scales.
            If None, default weights from the paper [1] will be used.
            Default weights: (0.0448, 0.2856, 0.3001, 0.2363, 0.1333).
        reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        data_range: The difference between the maximum and minimum of the pixel value,
            i.e., if for image x it holds min(x) = 0 and max(x) = 1, then data_range = 1.
            The pixel value interval of both input and output should remain the same.

    Shape:
        - Input: Required to be 2D (H, W), 3D (C,H,W), 4D (N,C,H,W) or 5D (N,C,H,W,2), channels first.
            The size of the image should be (kernel_size - 1) * 2 ** (levels - 1) + 1.
        - Target: Required to be 2D (H, W), 3D (C,H,W), 4D (N,C,H,W) or 5D (N,C,H,W,2), channels first.
            The size of the image should be (kernel_size - 1) * 2 ** (levels - 1) + 1.

    Examples::
        >>> loss = MultiScaleSSIMLoss()
        >>> input = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> target = torch.rand(3, 3, 256, 256)
        >>> output = loss(input, target)
        >>> output.backward()

    References:
        .. [1] Wang, Z., Simoncelli, E. P., Bovik, A. C. (2003).
           Multi-scale Structural Similarity for Image Quality Assessment.
           IEEE Asilomar Conference on Signals, Systems and Computers, 37,
           https://ieeexplore.ieee.org/document/1292216
           :DOI:`10.1109/ACSSC.2003.1292216`
        .. [2] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
           (2004). Image quality assessment: From error visibility to
           structural similarity. IEEE Transactions on Image Processing,
           13, 600-612.
           https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
           :DOI:`10.1109/TIP.2003.819861`
    """
    __constants__ = ['kernel_size', 'k1', 'k2', 'sigma', 'kernel', 'reduction']

    def __init__(self, kernel_size: int = 11, kernel_sigma: float = 1.5, k1: float = 0.01, k2: float = 0.03,
                 scale_weights: Optional[Union[Tuple[float], List[float], torch.Tensor]] = None,
                 reduction: str = 'mean', data_range: Union[int, float] = 1.) -> None:
        super().__init__()

        # Generic loss parameters.
        self.reduction = reduction

        # Loss-specific parameters.
        if scale_weights is None:
            scale_weights_from_ms_ssim_paper = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
            scale_weights = scale_weights_from_ms_ssim_paper
        self.scale_weights = scale_weights if isinstance(scale_weights, torch.Tensor) else torch.tensor(scale_weights)
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self.k1 = k1
        self.k2 = k2
        self.data_range = data_range

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Computation of Multi-scale Structural Similarity (MS-SSIM) index as a loss function.

        Args:
            prediction: Tensor of prediction of the network. Required to be
                2D (H, W), 3D (C,H,W), 4D (N,C,H,W) or 5D (N,C,H,W,2), channels first. The size of the image
                should be (kernel_size - 1) * 2 ** (levels - 1) + 1.
            target: Reference tensor. Required to be 2D (H, W), 3D (C,H,W), 4D (N,C,H,W) or 5D (N,C,H,W,2),
                channels first. The size of the image should be (kernel_size - 1) * 2 ** (levels - 1) + 1.

        Returns:
            Value of MS-SSIM loss to be minimized, i.e. 1-`ms_sim`. 0 <= MS-SSIM loss <= 1. In case of 5D tensor,
            complex value is returned as a tensor of size 2.
        """

        score = multi_scale_ssim(x=prediction, y=target, kernel_size=self.kernel_size, kernel_sigma=self.kernel_sigma,
                                 data_range=self.data_range, reduction=self.reduction, scale_weights=self.scale_weights,
                                 k1=self.k1, k2=self.k2)
        return torch.ones_like(score) - score


def _ssim_per_channel(x: torch.Tensor, y: torch.Tensor, kernel: torch.Tensor,
                      data_range: Union[float, int] = 1., k1: float = 0.01,
                      k2: float = 0.03) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Calculate Structural Similarity (SSIM) index for X and Y per channel.

    Args:
        x: Batch of images, (N,C,H,W).
        y: Batch of images, (N,C,H,W).
        kernel: 2D Gaussian kernel.
        data_range: Value range of input images (usually 1.0 or 255).
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Full Value of Structural Similarity (SSIM) index.
    """

    if x.size(-1) < kernel.size(-1) or x.size(-2) < kernel.size(-2):
        raise ValueError(f'Kernel size can\'t be greater than actual input size. Input size: {x.size()}. '
                         f'Kernel size: {kernel.size()}')

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    n_channels = x.size(1)
    mu1 = F.conv2d(x, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu2 = F.conv2d(y, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    compensation = 1.0
    sigma1_sq = compensation * (F.conv2d(x * x, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_sq)
    sigma2_sq = compensation * (F.conv2d(y * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu2_sq)
    sigma12 = compensation * (F.conv2d(x * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_mu2)

    # Set alpha = beta = gamma = 1.
    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map

    ssim_val = ssim_map.mean(dim=(-1, -2))
    cs = cs_map.mean(dim=(-1, -2))
    return ssim_val, cs


def _multi_scale_ssim(x: torch.Tensor, y: torch.Tensor, data_range: Union[int, float], kernel: torch.Tensor,
                      scale_weights_tensor: torch.Tensor, k1: float, k2: float) -> torch.Tensor:
    r"""Calculates Multi scale Structural Similarity (MS-SSIM) index for X and Y.

    Args:
        x: Batch of images, (N,C,H,W).
        y: Batch of images, (N,C,H,W).
        data_range: Value range of input images (usually 1.0 or 255).
        kernel: 2D Gaussian kernel.
        scale_weights_tensor: Weights for scaled SSIM
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Value of Multi scale Structural Similarity (MS-SSIM) index.
    """
    levels = scale_weights_tensor.size(0)
    min_size = (kernel.size(-1) - 1) * 2 ** (levels - 1) + 1
    if x.size(-1) < min_size or x.size(-2) < min_size:
        raise ValueError(f'Invalid size of the input images, expected at least {min_size}x{min_size}.')

    mcs = []
    ssim_val = None
    for iteration in range(levels):
        if iteration > 0:
            padding = (x.shape[2] % 2, x.shape[3] % 2)
            x = F.pad(x, pad=[padding[0], 0, padding[1], 0], mode='replicate')
            y = F.pad(y, pad=[padding[0], 0, padding[1], 0], mode='replicate')
            x = F.avg_pool2d(x, kernel_size=2, padding=0)
            y = F.avg_pool2d(y, kernel_size=2, padding=0)

        ssim_val, cs = _ssim_per_channel(x, y, kernel=kernel, data_range=data_range, k1=k1, k2=k2)
        mcs.append(cs)

    # mcs, (level, batch)
    mcs_ssim = torch.relu(torch.stack(mcs[:-1] + [ssim_val], dim=0))

    # weights, (level)
    msssim_val = torch.prod((mcs_ssim ** scale_weights_tensor.view(-1, 1, 1)), dim=0).mean(1)

    return msssim_val


def _ssim_per_channel_complex(x: torch.Tensor, y: torch.Tensor, kernel: torch.Tensor,
                              data_range: Union[float, int] = 1., k1: float = 0.01,
                              k2: float = 0.03) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Calculate Structural Similarity (SSIM) index for Complex X and Y per channel.

    Args:
        x: Batch of complex images, (N,C,H,W,2).
        y: Batch of complex images, (N,C,H,W,2).
        kernel: 2-D gauss kernel.
        data_range: Value range of input images (usually 1.0 or 255).
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Full Value of Complex Structural Similarity (SSIM) index.
    """
    n_channels = x.size(1)
    if x.size(-2) < kernel.size(-1) or x.size(-3) < kernel.size(-2):
        raise ValueError(f'Kernel size can\'t be greater than actual input size. Input size: {x.size()}. '
                         f'Kernel size: {kernel.size()}')

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    x_real = x[..., 0]
    x_imag = x[..., 1]
    y_real = y[..., 0]
    y_imag = y[..., 1]

    mu1_real = F.conv2d(x_real, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu1_imag = F.conv2d(x_imag, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu2_real = F.conv2d(y_real, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu2_imag = F.conv2d(y_imag, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu1_sq = mu1_real.pow(2) + mu1_imag.pow(2)
    mu2_sq = mu2_real.pow(2) + mu2_imag.pow(2)
    mu1_mu2_real = mu1_real * mu2_real - mu1_imag * mu2_imag
    mu1_mu2_imag = mu1_real * mu2_imag + mu1_imag * mu2_real

    compensation = 1.0

    x_sq = x_real.pow(2) + x_imag.pow(2)
    y_sq = y_real.pow(2) + y_imag.pow(2)
    x_y_real = x_real * y_real - x_imag * y_imag
    x_y_imag = x_real * y_imag + x_imag * y_real

    sigma1_sq = F.conv2d(x_sq, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_sq
    sigma2_sq = F.conv2d(y_sq, weight=kernel, stride=1, padding=0, groups=n_channels) - mu2_sq
    sigma12_real = F.conv2d(x_y_real, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_mu2_real
    sigma12_imag = F.conv2d(x_y_imag, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_mu2_imag
    sigma12 = torch.stack((sigma12_imag, sigma12_real), dim=-1)
    mu1_mu2 = torch.stack((mu1_mu2_real, mu1_mu2_imag), dim=-1)
    # Set alpha = beta = gamma = 1.
    cs_map = (sigma12 * 2 + c2 * compensation) / (sigma1_sq.unsqueeze(-1) + sigma2_sq.unsqueeze(-1) + c2 * compensation)
    ssim_map = (mu1_mu2 * 2 + c1 * compensation) / (mu1_sq.unsqueeze(-1) + mu2_sq.unsqueeze(-1) + c1 * compensation)
    ssim_map = ssim_map * cs_map

    ssim_val = ssim_map.mean(dim=(-2, -3))
    cs = cs_map.mean(dim=(-2, -3))

    return ssim_val, cs


def _multi_scale_ssim_complex(x: torch.Tensor, y: torch.Tensor, data_range: Union[int, float],
                              kernel: torch.Tensor, scale_weights_tensor: torch.Tensor, k1: float,
                              k2: float) -> torch.Tensor:
    r"""Calculate Multi scale Structural Similarity (MS-SSIM) index for Complex X and Y.

    Args:
        x: Batch of complex images, (N,C,H,W,2).
        y: Batch of complex images, (N,C,H,W,2).
        data_range: Value range of input images (usually 1.0 or 255).
        kernel: 2-D gauss kernel.
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Value of Complex Multi scale Structural Similarity (MS-SSIM) index.
    """
    levels = scale_weights_tensor.size(0)
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
            padding = (x.size(2) % 2, x.size(3) % 2)
            x_real = F.pad(x_real, pad=[padding[0], 0, padding[1], 0], mode='replicate')
            x_imag = F.pad(x_imag, pad=[padding[0], 0, padding[1], 0], mode='replicate')
            y_real = F.pad(y_real, pad=[padding[0], 0, padding[1], 0], mode='replicate')
            y_imag = F.pad(y_imag, pad=[padding[0], 0, padding[1], 0], mode='replicate')

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

    mcs_ssim_pow_abs = mcs_ssim_abs ** scale_weights_tensor.view(-1, 1, 1)
    mcs_ssim_pow_deg = mcs_ssim_deg * scale_weights_tensor.view(-1, 1, 1)

    msssim_val_abs = torch.prod(mcs_ssim_pow_abs, dim=0)
    msssim_val_deg = torch.sum(mcs_ssim_pow_deg, dim=0)
    msssim_val_real = msssim_val_abs * torch.cos(msssim_val_deg)
    msssim_val_imag = msssim_val_abs * torch.sin(msssim_val_deg)
    msssim_val = torch.stack((msssim_val_real, msssim_val_imag), dim=-1).mean(dim=1)
    return msssim_val
