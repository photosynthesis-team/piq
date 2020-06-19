r""" This module implements Peak Signal-to-Noise Ratio (PSNR) in PyTorch.
"""
import torch
from typing import Optional, Union

from piq.utils import _validate_input, _adjust_dimensions


def psnr(x: torch.Tensor, y: torch.Tensor, data_range: Union[int, float] = 1.0,
         reduction: Optional[str] = 'mean', convert_to_greyscale: bool = False):
    r"""Compute Peak Signal-to-Noise Ratio for a batch of images.
    Supports both greyscale and color images with RGB channel order.

    Args:
        x: Batch of predicted images with shape (batch_size x channels x H x W)
        y: Batch of target images with shape  (batch_size x channels x H x W)
        data_range: Value range of input images (usually 1.0 or 255). Default: 1.0
        reduction: Reduction over samples in batch: "mean"|"sum"|"none"
        convert_to_greyscale: Convert RGB image to YCbCr format and computes PSNR
            only on luminance channel if `True`. Compute on all 3 channels otherwise.
        
    Returns:
        PSNR: Index of similarity betwen two images.
    Note:
        Implementaition is based on Wikepedia https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    _validate_input((x, y), allow_5d=False)
    x, y = _adjust_dimensions(input_tensors=(x, y))

    # Constant for numerical stability
    EPS = 1e-8

    x = x / data_range
    y = y / data_range

    if (x.size(1) == 3) and convert_to_greyscale:
        # Convert RGB image to YCbCr and take luminance: Y = 0.299 R + 0.587 G + 0.114 B
        rgb_to_grey = torch.tensor([0.299, 0.587, 0.114]).view(1, -1, 1, 1).to(x)
        x = torch.sum(x * rgb_to_grey, dim=1, keepdim=True)
        y = torch.sum(y * rgb_to_grey, dim=1, keepdim=True)

    mse = torch.mean((x - y) ** 2, dim=[1, 2, 3])
    score = - 10 * torch.log10(mse + EPS)

    if reduction == 'none':
        return score

    return {'mean': score.mean,
            'sum': score.sum
            }[reduction](dim=0)
