r""" This module implements Information Weighted Structural Similarity Index Measure (IW-SSIM) using PyTorch.

It is based on original MATLAB code from authors [1] and PyTorch port by Jack Guo Xy [2].

References:
[1] https://ece.uwaterloo.ca/~z70wang/research/iwssim/iwssim_iwpsnr.zip
[2] https://github.com/Jack-guo-xy/Python-IW-SSIM

Author: @zakajd
"""

from typing import Iterable, Union, Optional

import torch.nn.functional as F
import pyrtools as pt
import numpy as np
import torch

from piq.functional import rgb2yiq, binomial_filter, average_filter

def information_weighted_ssim(x: torch.Tensor, y: torch.Tensor, kernel_size: int = 11, kernel_sigma: float = 1.5,
             data_range: Union[int, float] = 1., reduction: str = 'mean',
             scale_weights: torch.Tensor = None,
             k1: float = 0.01, k2: float = 0.03) -> torch.Tensor:
    r""" Interface of Information Weighted Structural Similarity (IW-SSIM) index.
    Inputs supposed to be in range [0, data_range] with RGB channels order for colour images.
    The size of the image should be at least (kernel_size - 1) * 2 ** (levels - 1) + 1.
    IW-SSIM is computed on greyscale images, so colour images are first converted into YIQ color space
    and only Y channel is used.

    Args:
        x: Tensor with shape 2D (H, W), 3D (C, H, W), 4D (N, C, H, W).
        y: Tensor with shape 2D (H, W), 3D (C, H, W), 4D (N, C, H, W).
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
        Value of Information Weighted Structural Similarity (IW-SSIM) index.
    References:
        .. [1] Wang, Z., Li Q. (2011).
           Information Content Weighting for Perceptual Image Quality Assessment.
           IEEE TRANSACTIONS ON IMAGE PROCESSING, VOL. 20, NO. 5
           https://ece.uwaterloo.ca/~z70wang/publications/IWSSIM.pdf

        .. [2] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
           (2004). Image quality assessment: From error visibility to
           structural similarity. IEEE Transactions on Image Processing,
           13, 600-612.
           https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
           :DOI:`10.1109/TIP.2003.819861`
    """
    _validate_input(
        input_tensors=(x, y), data_range=data_range)
    x, y = _adjust_dimensions(input_tensors=(x, y))
    
    # Scale to [0, 255] range to match constants
    x = x / data_range * 255
    y = y / data_range * 255

    if scale_weights is None:
        # Values from IW-SSIM paper
        scale_weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=x.device)

    # Convert RGB to YIQ color space https://en.wikipedia.org/wiki/YIQ and take Y channel
    if x.size(1) == 3:
        x = rgb2yiq(x)[:, : 1]
        y = rgb2yiq(y)[:, : 1]

    levels = scale_weights.size(0)
    min_size = kernel_size * 2 ** (levels - 1) + 1
    if x.size(-1) < min_size or x.size(-2) < min_size:
        raise ValueError(f'Invalid size of the input images, expected at least {min_size}x{min_size}.')
    
    # Used for Laplacian pyramid construction
    PYRAMID_KERNEL_SIZE = 5
    pyramid_kernel = binomial_filter(PYRAMID_KERNEL_SIZE).repeat(x.size(1), 1, 1, 1).to(x)
    
    # Used for "bag of nails" upsampling
    upsample_kernel = torch.tensor([[[[1., 0.], [0., 0.]]]]).to(x)
    
    # Used for SSIM computation
    kernel = gaussian_filter(kernel_size, kernel_sigma).repeat(x.size(1), 1, 1, 1).to(x)
    
    wmcs = []
    for i in range(levels):  # Last level is downsampled image
        # Valid padding
        up_pad = PYRAMID_KERNEL_SIZE // 2
        down_pad = PYRAMID_KERNEL_SIZE // 2 + max(x.shape[2] % 2, x.shape[3] % 2)
        pad_to_use = [up_pad, down_pad, up_pad, down_pad]
        
        # Blur and downsample
        x_down = F.conv2d(F.pad(x, pad=pad_to_use, mode='reflect'), pyramid_kernel)[..., ::2, ::2]
        y_down = F.conv2d(F.pad(x, pad=pad_to_use, mode='reflect'), pyramid_kernel)[..., ::2, ::2]
        
        # Upsample and blur
        x_up = F.conv_transpose2d(x_down, upsample_kernel, stride=2, padding=0)
        x_up_blured = F.conv2d(F.pad(x_up, pad=pad_to_use, mode='reflect'), kernel * 4)
        
        y_up = F.conv_transpose2d(y_down, upsample_kernel, stride=2, padding=0)
        y_up_blured = F.conv2d(F.pad(y_up, pad=pad_to_use, mode='reflect'), kernel * 4)
        
        x_diff = x - x_up_blured
        y_diff = y - y_up_blured
        
        # For last level compute SSIM between images, not their diffs.
        ssim_map, cs_map = _ssim_per_channel(
            x=x if i == (level - 1) else x_diff,
            y=y if i == (level - 1) else y_diff,
            kernel=kernel,
            data_range=data_range,
            k1=k1,
            k2=k2
        )
    
        iw_map = info_content_weight(
            x=x if i == (level - 1) else x_diff,
            y=y if i == (level - 1) else y_diff,
        )

        # Average among spatial dimensions
        ssim_val = (ssim_map * iw_map).sum(dim=(2, 3)) / iw_map.sum(dim=(2, 3))
        wcs = (cs_map * iw_map).sum(dim=(2, 3)) / iw_map.sum(dim=(2, 3))
        wmcs.append(wcs)
        
    # wmcs, (level, batch) ## Use torch.abs??
    wmcs_ssim = torch.relu(torch.stack(wmcs[:-1] + [ssim_val], dim=0))

    # weights, (level)
    iwssim_val = torch.prod((wmcs_ssim ** scale_weights.view(-1, 1, 1).to(x)), dim=0).mean(1)

    if reduction == 'none':
        return iwssim_val

    return {'mean': torch.mean,
            'sum': torch.sum}[reduction](iwssim_val, dim=0)


def info_content_weight(x: torch.Tensor, y: torch.Tensor, kernel_size: int = 3):
    """Compute information content weightning between 2 tensors.
    Algorithms is almost identical to the one described in the paper.
    One differce is that no parent subbands are taken into account.
    
    Args:
        x: Tensor with shape (N, C, H, W).
        y: Tensor with shape (N, C, H, W).

    References:
        .. [1] Wang, Z., Li Q. (2011).
           Information Content Weighting for Perceptual Image Quality Assessment.
           IEEE TRANSACTIONS ON IMAGE PROCESSING, VOL. 20, NO. 5
           https://ece.uwaterloo.ca/~z70wang/publications/IWSSIM.pdf
    """
    EPS = torch.finfo(x.dtype).eps

    kernel = average_filter(kernel_size)
    padding = int((kernel_size - 1) / 2)

    # TODO: Check if padding really needed?
    # Prepare for estimating IW-SSIM parameters
    mu_x, mu_y = F.conv2d(x, kernel), F.conv2d(y, kernel)  # valid padding
    mu_x_sq, mu_y_sq, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x_sq = F.conv2d(x ** 2, kernel) - mu_x_sq
    sigma_y_sq = F.conv2d(y ** 2, kernel) - mu_y_sq
    sigma_xy = F.conv2d(x * y, kernel) - mu_xy

    # Zero small negative values
    sigma_x_sq = torch.relu(sigma_x_sq)
    sigma_y_sq = torch.relu(sigma_y_sq)

    # Estimate gain factor and error
    g = sigma_xy / (sigma_y_sq + EPS)
    sigma_v_sq = sigma_x_sq - g * sigma_xy

    g = g.masked_fill(sigma_y_sq < EPS, 0)
    sigma_v_sq[sigma_y_sq < EPS] = sigma_x_sq[sigma_y_sq < EPS]
    sigma_y_sq = sigma_y_sq.masked_fill(sigma_y_sq < EPS, 0)
    g = g.masked_fill(sigma_x_sq < EPS, 0)
    sigma_v_sq = sigma_v_sq.masked_fill(sigma_x_sq < EPS, 0)


    return None

class InformationWeightedSSIMLoss():
    pass