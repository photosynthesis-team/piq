r"""Filters for gradient computation, bluring, etc."""
import torch
import numpy as np
from typing import Optional


def haar_filter(kernel_size: int, device: Optional[str] = None, dtype: Optional[type] = None) -> torch.Tensor:
    r"""Creates Haar kernel

    Args:
        kernel_size: size of the kernel
        device: target device for kernel generation
        dtype: target data type for kernel generation
    Returns:
        kernel: Tensor with shape (1, kernel_size, kernel_size)
    """
    kernel = torch.ones((kernel_size, kernel_size), device=device, dtype=dtype) / kernel_size
    kernel[kernel_size // 2:, :] = - kernel[kernel_size // 2:, :]
    return kernel.unsqueeze(0)


def hann_filter(kernel_size: int, device: Optional[str] = None, dtype: Optional[type] = None) -> torch.Tensor:
    r"""Creates  Hann kernel
    Args:
        kernel_size: size of the kernel
        device: target device for kernel generation
        dtype: target data type for kernel generation
    Returns:
        kernel: Tensor with shape (1, kernel_size, kernel_size)
    """
    # Take bigger window and drop borders
    window = torch.hann_window(kernel_size + 2, periodic=False, device=device, dtype=dtype)[1:-1]
    kernel = window[:, None] * window[None, :]
    # Normalize and reshape kernel
    return kernel.view(1, kernel_size, kernel_size) / kernel.sum()


def gaussian_filter(kernel_size: int, sigma: float, device: Optional[str] = None,
                    dtype: Optional[type] = None) -> torch.Tensor:
    r"""Returns 2D Gaussian kernel N(0,`sigma`^2)
    Args:
        size: Size of the kernel
        sigma: Std of the distribution
        device: target device for kernel generation
        dtype: target data type for kernel generation
    Returns:
        gaussian_kernel: Tensor with shape (1, kernel_size, kernel_size)
    """
    coords = torch.arange(kernel_size, dtype=dtype, device=device)
    coords -= (kernel_size - 1) / 2.

    g = coords ** 2
    g = (- (g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma ** 2)).exp()

    g /= g.sum()
    return g.unsqueeze(0)


# Gradient operator kernels
def scharr_filter(device: Optional[str] = None, dtype: Optional[type] = None) -> torch.Tensor:
    r"""Utility function that returns a normalized 3x3 Scharr kernel in X direction

    Args:
        device: target device for kernel generation
        dtype: target data type for kernel generation
    Returns:
        kernel: Tensor with shape (1, 3, 3)
    """
    return torch.tensor([[[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]]], device=device, dtype=dtype) / 16


def prewitt_filter(device: Optional[str] = None, dtype: Optional[type] = None) -> torch.Tensor:
    r"""Utility function that returns a normalized 3x3 Prewitt kernel in X direction

    Args:
        device: target device for kernel generation
        dtype: target data type for kernel generation
    Returns:
        kernel: Tensor with shape (1, 3, 3)"""
    return torch.tensor([[[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]]], device=device, dtype=dtype) / 3


def binomial_filter1d(kernel_size: int, device: Optional[str] = None, dtype: Optional[type] = None) -> torch.Tensor:
    r"""Creates 1D normalized binomial filter

    Args:
        kernel_size (int): kernel size
        device: target device for kernel generation
        dtype: target data type for kernel generation

    Returns:
        Binomial kernel with shape (1, 1, kernel_size)
    """
    kernel = np.poly1d([0.5, 0.5]) ** (kernel_size - 1)
    return torch.tensor(kernel.c, dtype=dtype, device=device).view(1, 1, kernel_size)


def average_filter2d(kernel_size: int, device: Optional[str] = None, dtype: Optional[type] = None) -> torch.Tensor:
    r"""Creates 2D normalized average filter

    Args:
        kernel_size (int): kernel size
        device: target device for kernel generation
        dtype: target data type for kernel generation

    Returns:
        kernel: Tensor with shape (1, kernel_size, kernel_size)
    """
    window = torch.ones(kernel_size, dtype=dtype, device=device) / kernel_size
    kernel = window[:, None] * window[None, :]
    return kernel.unsqueeze(0)
