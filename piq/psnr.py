r""" This module implements Peak Signal-to-Noise Ratio (PSNR) in PyTorch.
"""
import torch
from typing import Union

from piq.utils import _validate_input, _reduce


def psnr(x: torch.Tensor, y: torch.Tensor, data_range: Union[int, float] = 1.0,
         reduction: str = 'mean', convert_to_greyscale: bool = False) -> torch.Tensor:
    r"""Compute Peak Signal-to-Noise Ratio for a batch of images.
    Supports both greyscale and color images with RGB channel order.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        data_range: Maximum value range of images (usually 1.0 or 255).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        convert_to_greyscale: Convert RGB image to YIQ format and computes PSNR
            only on luminance channel if `True`. Compute on all 3 channels otherwise.

    Returns:
        PSNR Index of similarity between two images.

    References:
        https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    _validate_input([x, y], dim_range=(4, 5), data_range=(0, data_range))

    # Constant for numerical stability
    EPS = 1e-8

    x = x / float(data_range)
    y = y / float(data_range)

    if (x.size(1) == 3) and convert_to_greyscale:
        # Convert RGB image to YIQ and take luminance: Y = 0.299 R + 0.587 G + 0.114 B
        rgb_to_grey = torch.tensor([0.299, 0.587, 0.114], device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        x = torch.sum(x * rgb_to_grey, dim=1, keepdim=True)
        y = torch.sum(y * rgb_to_grey, dim=1, keepdim=True)

    mse = torch.mean((x - y) ** 2, dim=[1, 2, 3])
    score: torch.Tensor = - 10 * torch.log10(mse + EPS)

    return _reduce(score, reduction)
