r"""General purpose functions"""
from typing import Tuple
import torch


def ifftshift(x: torch.Tensor) -> torch.Tensor:
    r""" Similar to np.fft.ifftshift but applies to PyTorch Tensors"""
    shift = [-(ax // 2) for ax in x.size()]
    return torch.roll(x, shift, tuple(range(len(shift))))


def get_meshgrid(size: Tuple[int, int]) -> torch.Tensor:
    r"""Return coordinate grid matrices centered at zero point.
    Args:
        size: Shape of meshgrid to create
    """
    if size[0] % 2:
        # Odd
        x = torch.arange(-(size[0] - 1) / 2, size[0] / 2) / (size[0] - 1)
    else:
        # Even
        x = torch.arange(- size[0] / 2, size[0] / 2) / size[0]

    if size[1] % 2:
        # Odd
        y = torch.arange(-(size[1] - 1) / 2, size[1] / 2) / (size[1] - 1)
    else:
        # Even
        y = torch.arange(- size[1] / 2, size[1] / 2) / size[1]
    return torch.meshgrid(x, y)


def similarity_map(map_x: torch.Tensor, map_y: torch.Tensor, constant: float) -> torch.Tensor:
    r""" Compute similarity_map between two tensors using Dice-like equation.

    Args:
        map_x: Tensor with map to be compared
        map_y: Tensor with map to be compared
        constant: Used for numerical stability
    """
    return (2.0 * map_x * map_y + constant) / (map_x ** 2 + map_y ** 2 + constant)


def gradient_map(x: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
    r""" Compute gradient map for a given tensor and stack of kernels.

    Args:
        x: Tensor with shape B x C x H x W
        kernels: Stack of tensors for gradient computation with shape N x k_W x k_H
    Returns:
        Gradients of x per-channel with shape B x C x H x W
    """
    padding = kernels.size(-1) // 2
    grads = torch.nn.functional.conv2d(x, kernels.to(x), padding=padding)

    return torch.sqrt(torch.sum(grads ** 2, dim=-3, keepdim=True))
