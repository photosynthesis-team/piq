r"""Filters for gradient computation, bluring, etc."""
import torch
import numpy as np


def haar_filter(kernel_size: int) -> torch.Tensor:
    r"""Creates Haar kernel
    Returns:
        kernel: Tensor with shape (1, kernel_size, kernel_size)
    """
    kernel = torch.ones((kernel_size, kernel_size)) / kernel_size
    kernel[kernel_size // 2:, :] = - kernel[kernel_size // 2:, :]
    return kernel.unsqueeze(0)


def hann_filter(kernel_size: int) -> torch.Tensor:
    r"""Creates  Hann kernel
    Returns:
        kernel: Tensor with shape (1, kernel_size, kernel_size)
    """
    # Take bigger window and drop borders
    window = torch.hann_window(kernel_size + 2, periodic=False)[1:-1]
    kernel = window[:, None] * window[None, :]
    # Normalize and reshape kernel
    return kernel.view(1, kernel_size, kernel_size) / kernel.sum()


def gaussian_filter(kernel_size: int, sigma: float, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    r"""Returns 2D Gaussian kernel N(0,`sigma`^2)
    Args:
        size: Size of the kernel
        sigma: Std of the distribution
        dtype: type of tensor to return
    Returns:
        gaussian_kernel: Tensor with shape (1, kernel_size, kernel_size)
    """
    coords = torch.arange(kernel_size, dtype=dtype)
    coords -= (kernel_size - 1) / 2.

    g = coords ** 2
    g = (- (g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma ** 2)).exp()

    g /= g.sum()
    return g.unsqueeze(0)


# Gradient operator kernels
def scharr_filter() -> torch.Tensor:
    r"""Utility function that returns a normalized 3x3 Scharr kernel in X direction
    Returns:
        kernel: Tensor with shape (1, 3, 3)
    """
    return torch.tensor([[[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]]]) / 16


def prewitt_filter() -> torch.Tensor:
    r"""Utility function that returns a normalized 3x3 Prewitt kernel in X direction
    Returns:
        kernel: Tensor with shape (1, 3, 3)"""
    return torch.tensor([[[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]]]) / 3


def binomial_filter1d(kernel_size: int) -> torch.Tensor:
    r"""Creates 1D normalized binomial filter

    Args:
        kernel_size (int): kernel size

    Returns:
        Binomial kernel with shape (1, 1, kernel_size)
    """
    kernel = np.poly1d([0.5, 0.5]) ** (kernel_size - 1)
    return torch.tensor(kernel.c).view(1, 1, kernel_size)


def average_filter2d(kernel_size: int) -> torch.Tensor:
    r"""Creates 2D normalized average filter

    Args:
        kernel_size (int):

    Returns:
        kernel: Tensor with shape (1, kernel_size, kernel_size)
    """
    window = torch.ones(kernel_size) / kernel_size
    kernel = window[:, None] * window[None, :]
    return kernel.unsqueeze(0)
