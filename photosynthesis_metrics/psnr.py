r""" This module implements Peak Signal-to-Noise Ration (PSNR) in PyTorch.
"""
import torch
from typing import Optional, Union

from photosynthesis_metrics.utils import _validate_input


def psnr(prediction: torch.Tensor, target: torch.Tensor,
         data_range: Union[int, float] = 1.0, reduction: Optional[str] = 'mean'):
    r"""Compute Peak Signal-to-Noise Ration for a batch of images.
    Supports both greyscale and color images with RGB channel order.

    Args:
        prediction: Batch of predicted images with shape (batch_size x channels x H x W)
        target: Batch of target images with shape  (batch_size x channels x H x W)
        data_range: Value range of input images (usually 1.0 or 255). Default: 1.0
        reduction: Reduction over samples in batch: "mean"|"sum"|"none"
        
    Returns:
        PSNR: Index of similarity betwen two images.
    Note:
        Implementaition is based on Wikepedia https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        Colour images a first converted to YCbCr format and only luminance component is considered.
    """
    _validate_input((prediction, target), allow_5d=False)

    # Constant for numerical stability
    EPS = 1e-8

    if data_range == 255:
        prediction = prediction / 255.
        target = target / 255.

    # Convert RGB image to YCbCr and take luminance: Y = 0.299 R + 0.587 G + 0.114 B
    num_channels = prediction.size(1)
    if num_channels == 3:
        prediction = 0.299 * prediction[:, 0, :, :] + 0.587 * prediction[:, 1, :, :] + 0.114 * prediction[:, 2, :, :]
        target = 0.299 * target[:, 0, :, :] + 0.587 * target[:, 1, :, :] + 0.114 * target[:, 2, :, :]

        # Add channel dimension
        prediction = prediction[:, None, :, :]
        target = target[:, None, :, :]

    mse = torch.mean((prediction - target) ** 2, dim=[1, 2, 3])
    psnr = - 10 * torch.log10(mse + EPS)

    if reduction == 'mean':
        return psnr.mean(dim=0)
    elif reduction == 'sum':
        return psnr.sum(dim=0)
    elif reduction == 'none':
        return psnr
    else:
        raise ValueError(f'Expected reduction modes are "mean"|"sum"|"none", got {reduction}')
