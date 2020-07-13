import torch


def hann_filter(kernel_size) -> torch.Tensor:
    r"""Creates  Hann kernel
    Returns:
        kernel: Tensor with shape 1 x `kernel_size` x `kernel_size`"""
    # Take bigger window and drop borders
    window = torch.hann_window(kernel_size + 2, periodic=False)[1:-1]
    kernel = window[:, None] * window[None, :]
    # Normalize and reshape kernel
    return kernel.view(1, kernel_size, kernel_size) / kernel.sum()


# Gradient operator kernels
def scharr_filter() -> torch.Tensor:
    r"""Utility function that returns a normalized 3x3 Scharr kernel in X direction
    Returns:
        kernel: Tensor with shape 1x3x3"""
    return torch.tensor([[[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]]]) / 16


def prewitt_filter() -> torch.Tensor:
    r"""Utility function that returns a normalized 3x3 Prewitt kernel in X direction
    Returns:
        kernel: Tensor with shape 1x3x3"""
    return torch.tensor([[[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]]]) / 3
