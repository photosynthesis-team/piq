""" This module implements Structural Similarity (SSIM) index in PyTorch.

Implementation of classes and functions from this module are inspired by Gongfan Fang's implementation:
https://github.com/VainF/pytorch-msssim
"""
import torch

import torch.nn.functional as F

from typing import Union, Optional, Tuple, List


def _fspecial_gauss_1d(size: int, sigma: float) -> torch.Tensor:
    """ Creates a 1-D gauss kernel.

    Args:
        size: The size of gauss kernel.
        sigma: Sigma of normal distribution.
    Returns:
        A 1D kernel.
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size//2

    g = torch.exp(-(coords**2) / (2*sigma**2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(to_blur: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """ Blur input with 1-D kernel.

    Args:
        to_blur: A batch of tensors to be blured.
        window: 1-D gauss kernel.
    Returns:
        A batch of blurred tensors.
    """
    _, n_channels, _, _ = to_blur.shape
    out = F.conv2d(to_blur, window, stride=1, padding=0, groups=n_channels)
    out = F.conv2d(out, window.transpose(2, 3), stride=1, padding=0, groups=n_channels)
    return out


def compute_ssim(x: torch.Tensor, y: torch.Tensor, win: torch.Tensor, data_range: Union[float, int] = 255,
                 size_average: bool = True, full: bool = False,
                 k: Union[List[float, float], Tuple[float, float]] = (0.01, 0.03)) -> torch.Tensor:
    """Calculate Structural Similarity (SSIM) index for X and Y.

    Args:
        x: Batch of images, (N,C,H,W).
        y: Batch of images, (N,C,H,W).
        win: 1-D gauss kernel.
        data_range: Value range of input images (usually 1.0 or 255).
        size_average: If size_average=True, ssim of all images will be averaged as a scalar.
        full: Return sc or not.
        k: Constants used to avoid problems with negative covariances of input images.
    Returns:
        Value of Structural Similarity (SSIM) index.
    """
    k1, k2 = k

    c1 = (k1 * data_range)**2
    c2 = (k2 * data_range)**2

    win = win.to(x.device, dtype=x.dtype)

    mu1 = gaussian_filter(x, win)
    mu2 = gaussian_filter(y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    compensation = 1.0
    sigma1_sq = compensation * (gaussian_filter(x * x, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(y * y, win) - mu2_sq)
    sigma12   = compensation * (gaussian_filter(x * y, win) - mu1_mu2)

    # Set alpha = beta = gamma = 1.
    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map

    if size_average:
        ssim_val = ssim_map.mean()
        cs = cs_map.mean()
    else:
        # Reduce along CHW.
        ssim_val = ssim_map.mean(-1).mean(-1).mean(-1)
        cs = cs_map.mean(-1).mean(-1).mean(-1)

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def structural_similarity(x: torch.Tensor, y: torch.Tensor, win_size: int = 11, win_sigma: float = 1.5,
                          win: Optional[torch.Tensor] = None, data_range: Union[int, float] = 255,
                          size_average: bool = True, full: bool = False,
                          k: Union[List[float, float], Tuple[float, float]] = (0.01, 0.03)) -> torch.Tensor:
    """Interface of Structural Similarity (SSIM) index.

    Args:
        x: Batch of images, (N,C,H,W).
        y: Batch of images, (N,C,H,W).
        win_size: The size of gauss kernel.
        win_sigma: Sigma of normal distribution.
        win: 1-D gauss kernel. If None, a new kernel will be created according to win_size and win_sigma.
        data_range: Value range of input images (usually 1.0 or 255).
        size_average: If size_average=True, ssim of all images will be averaged as a scalar.
        full: Return sc or not.
        k: scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        Value of Structural Similarity (SSIM) index.
    """
    if len(x.shape) != 4:
        raise ValueError('Input images must be 4-d tensors.')

    if not x.type() == y.type():
        raise ValueError('Input images must have the same dtype.')

    if not x.shape == y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(x.shape[1], 1, 1, 1)

    ssim_val, cs = compute_ssim(x, y,
                                win=win,
                                data_range=data_range,
                                size_average=False,
                                full=True, k=k)
    if size_average:
        ssim_val = ssim_val.mean()
        cs = cs.mean()

    if full:
        return ssim_val, cs
    else:
        return ssim_val
