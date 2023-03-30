r"""Implementation of Visual Saliency-induced Index
Code is based on the MATLAB version for computations in pixel domain
https://sse.tongji.edu.cn/linzhang/IQA/VSI/VSI.htm

References:
    https://ieeexplore.ieee.org/document/6873260
"""
import warnings
import functools
from typing import Union

import torch
from torch.nn.modules.loss import _Loss
from torch.nn.functional import avg_pool2d, interpolate, pad

from piq.functional import ifftshift, gradient_map, scharr_filter, rgb2lmn, rgb2lab, similarity_map, get_meshgrid
from piq.utils import _validate_input, _reduce, _parse_version


def vsi(x: torch.Tensor, y: torch.Tensor, reduction: str = 'mean', data_range: Union[int, float] = 1.,
        c1: float = 1.27, c2: float = 386., c3: float = 130., alpha: float = 0.4, beta: float = 0.02,
        omega_0: float = 0.021, sigma_f: float = 1.34, sigma_d: float = 145., sigma_c: float = 0.001) -> torch.Tensor:
    r"""Compute Visual Saliency-induced Index for a batch of images.

    Both inputs are supposed to have RGB channels order in accordance with the original approach.
    Nevertheless, the method supports greyscale images, which they are converted to RGB by copying the grey
    channel 3 times.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        data_range: Maximum value range of images (usually 1.0 or 255).
        c1: coefficient to calculate saliency component of VSI
        c2: coefficient to calculate gradient component of VSI
        c3: coefficient to calculate color component of VSI
        alpha: power for gradient component of VSI
        beta: power for color component of VSI
        omega_0: coefficient to get log Gabor filter at SDSP
        sigma_f: coefficient to get log Gabor filter at SDSP
        sigma_d: coefficient to get SDSP
        sigma_c: coefficient to get SDSP

    Returns:
        Index of similarity between two images. Usually in [0, 1] range.

    References:
        L. Zhang, Y. Shen and H. Li, "VSI: A Visual Saliency-Induced Index for Perceptual Image Quality Assessment,"
        IEEE Transactions on Image Processing, vol. 23, no. 10, pp. 4270-4281, Oct. 2014, doi: 10.1109/TIP.2014.2346028
        https://ieeexplore.ieee.org/document/6873260

    Note:
        The original method supports only RGB image.
        See https://ieeexplore.ieee.org/document/6873260 for details.
    """
    _validate_input([x, y], dim_range=(4, 4), data_range=(0, data_range))

    if x.size(1) == 1:
        x = x.repeat(1, 3, 1, 1)
        y = y.repeat(1, 3, 1, 1)
        warnings.warn('The original VSI supports only RGB images. The input images were converted to RGB by copying '
                      'the grey channel 3 times.')

    # Scale to [0, 255] range to match scale of constant
    x = x * 255. / float(data_range)
    y = y * 255. / float(data_range)

    vs_x = sdsp(x, data_range=255, omega_0=omega_0,
                sigma_f=sigma_f, sigma_d=sigma_d, sigma_c=sigma_c)
    vs_y = sdsp(y, data_range=255, omega_0=omega_0, sigma_f=sigma_f,
                sigma_d=sigma_d, sigma_c=sigma_c)

    # Convert to LMN colour space
    x_lmn = rgb2lmn(x)
    y_lmn = rgb2lmn(y)

    # Averaging image if the size is large enough
    kernel_size = max(1, round(min(vs_x.size()[-2:]) / 256))
    padding = kernel_size // 2

    if padding:
        upper_pad = padding
        bottom_pad = (kernel_size - 1) // 2
        pad_to_use = [upper_pad, bottom_pad, upper_pad, bottom_pad]
        mode = 'replicate'
        vs_x = pad(vs_x, pad=pad_to_use, mode=mode)
        vs_y = pad(vs_y, pad=pad_to_use, mode=mode)
        x_lmn = pad(x_lmn, pad=pad_to_use, mode=mode)
        y_lmn = pad(y_lmn, pad=pad_to_use, mode=mode)

    vs_x = avg_pool2d(vs_x, kernel_size=kernel_size)
    vs_y = avg_pool2d(vs_y, kernel_size=kernel_size)

    x_lmn = avg_pool2d(x_lmn, kernel_size=kernel_size)
    y_lmn = avg_pool2d(y_lmn, kernel_size=kernel_size)

    # Calculate gradient map
    sch_filter = scharr_filter(device=x_lmn.device, dtype=x_lmn.dtype)
    kernels = torch.stack([sch_filter, sch_filter.transpose(1, 2)])
    gm_x = gradient_map(x_lmn[:, :1], kernels)
    gm_y = gradient_map(y_lmn[:, :1], kernels)

    # Calculate all similarity maps
    s_vs = similarity_map(vs_x, vs_y, c1)
    s_gm = similarity_map(gm_x, gm_y, c2)
    s_m = similarity_map(x_lmn[:, 1:2], y_lmn[:, 1:2], c3)
    s_n = similarity_map(x_lmn[:, 2:], y_lmn[:, 2:], c3)
    s_c = s_m * s_n

    s_c_complex = [s_c.abs(), torch.atan2(torch.zeros_like(s_c), s_c)]
    s_c_complex_pow = [s_c_complex[0] ** beta, s_c_complex[1] * beta]
    s_c_real_pow = s_c_complex_pow[0] * torch.cos(s_c_complex_pow[1])

    s = s_vs * s_gm.pow(alpha) * s_c_real_pow
    vs_max = torch.max(vs_x, vs_y)

    eps = torch.finfo(vs_max.dtype).eps
    output = s * vs_max
    output = ((output.sum(dim=(-1, -2)) + eps) / (vs_max.sum(dim=(-1, -2)) + eps)).squeeze(-1)

    return _reduce(output, reduction)


class VSILoss(_Loss):
    r"""Creates a criterion that measures Visual Saliency-induced Index error between
    each element in the input and target.

    The sum operation still operates over all the elements, and divides by :math:`n`.

    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

    Args:
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        data_range: Maximum value range of images (usually 1.0 or 255).
        c1: coefficient to calculate saliency component of VSI
        c2: coefficient to calculate gradient component of VSI
        c3: coefficient to calculate color component of VSI
        alpha: power for gradient component of VSI
        beta: power for color component of VSI
        omega_0: coefficient to get log Gabor filter at SDSP
        sigma_f: coefficient to get log Gabor filter at SDSP
        sigma_d: coefficient to get SDSP
        sigma_c: coefficient to get SDSP

    Examples:

        >>> loss = VSILoss()
        >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> y = torch.rand(3, 3, 256, 256)
        >>> output = loss(x, y)
        >>> output.backward()

    References:
        L. Zhang, Y. Shen and H. Li, "VSI: A Visual Saliency-Induced Index for Perceptual Image Quality Assessment,"
        IEEE Transactions on Image Processing, vol. 23, no. 10, pp. 4270-4281, Oct. 2014, doi: 10.1109/TIP.2014.2346028
        https://ieeexplore.ieee.org/document/6873260
    """

    def __init__(self, reduction: str = 'mean', c1: float = 1.27, c2: float = 386., c3: float = 130.,
                 alpha: float = 0.4, beta: float = 0.02, data_range: Union[int, float] = 1.,
                 omega_0: float = 0.021, sigma_f: float = 1.34, sigma_d: float = 145., sigma_c: float = 0.001) -> None:
        super().__init__()
        self.reduction = reduction
        self.data_range = data_range

        self.vsi = functools.partial(
            vsi, c1=c1, c2=c2, c3=c3, alpha=alpha, beta=beta, omega_0=omega_0,
            sigma_f=sigma_f, sigma_d=sigma_d, sigma_c=sigma_c, data_range=data_range, reduction=reduction)

    def forward(self, x, y):
        r"""Computation of VSI as a loss function.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Value of VSI loss to be minimized in [0, 1] range.

        Note:
            Both inputs are supposed to have RGB channels order in accordance with the original approach.
            Nevertheless, the method supports greyscale images, which they are converted to RGB by copying the grey
            channel 3 times.
        """

        return 1. - self.vsi(x=x, y=y)


def sdsp(x: torch.Tensor, data_range: Union[int, float] = 255, omega_0: float = 0.021, sigma_f: float = 1.34,
         sigma_d: float = 145., sigma_c: float = 0.001) -> torch.Tensor:
    r"""SDSP algorithm for salient region detection from a given image.

    Supports only colour images with RGB channel order.

    Args:
        x: Tensor. Shape :math:`(N, 3, H, W)`.
        data_range: Maximum value range of images (usually 1.0 or 255).
        omega_0: coefficient for log Gabor filter
        sigma_f: coefficient for log Gabor filter
        sigma_d: coefficient for the central areas, which have a bias towards attention
        sigma_c: coefficient for the warm colors, which have a bias towards attention

    Returns:
        torch.Tensor: Visual saliency map
    """
    x = x / float(data_range) * 255
    size = x.size()
    size_to_use = (256, 256)
    x = interpolate(input=x, size=size_to_use, mode='bilinear', align_corners=False)

    x_lab = rgb2lab(x, data_range=255)

    xx, yy = get_meshgrid(size_to_use, device=x_lab.device, dtype=x_lab.dtype)
    lg = _log_gabor(xx, yy, omega_0, sigma_f).view(1, 1, *size_to_use)

    recommended_torch_version = _parse_version('1.8.0')
    torch_version = _parse_version(torch.__version__)
    if len(torch_version) != 0 and torch_version >= recommended_torch_version:
        x_fft = torch.fft.fft2(x_lab)
        x_ifft_real = torch.fft.ifft2(x_fft * lg).real
    else:
        x_fft = torch.rfft(x_lab, 2, onesided=False)
        x_ifft_real = torch.ifft(x_fft * lg.unsqueeze(-1), 2)[..., 0]

    s_f = x_ifft_real.pow(2).sum(dim=1, keepdim=True).sqrt()

    coordinates = torch.stack([xx, yy], dim=0)
    coordinates = coordinates * size_to_use[0] + 1
    s_d = torch.exp(-torch.sum(coordinates ** 2, dim=0) / sigma_d ** 2).view(1, 1, *size_to_use)

    eps = torch.finfo(x_lab.dtype).eps
    min_x = x_lab.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values
    max_x = x_lab.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values
    normalized = (x_lab - min_x) / (max_x - min_x + eps)

    norm = normalized[:, 1:].pow(2).sum(dim=1, keepdim=True)
    s_c = 1 - torch.exp(-norm / sigma_c ** 2)

    vs_m = s_f * s_d * s_c
    vs_m = interpolate(vs_m, size[-2:], mode='bilinear', align_corners=True)
    min_vs_m = vs_m.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values
    max_vs_m = vs_m.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values
    return (vs_m - min_vs_m) / (max_vs_m - min_vs_m + eps)


def _log_gabor(xx: torch.Tensor, yy: torch.Tensor, omega_0: float, sigma_f: float, ) -> torch.Tensor:
    r"""Creates log Gabor filter

    Args:
        xx: x component of meshgrid
        yy: y component of meshgrid
        omega_0: center frequency of the filter
        sigma_f: bandwidth of the filter
        device: target device for kernel generation
        dtype: target data type for kernel generation
    Returns:
        log Gabor filter
    """
    radius = (xx ** 2 + yy ** 2).sqrt()
    mask = radius <= 0.5

    r = radius * mask
    r = ifftshift(r)
    r[0, 0] = 1

    lg = torch.exp((-(r / omega_0).log().pow(2)) / (2 * sigma_f ** 2))
    lg[0, 0] = 0
    return lg
