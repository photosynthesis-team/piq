r"""General purpose functions"""
from typing import Tuple, Union
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


def similarity_map(map_x: torch.Tensor, map_y: torch.Tensor, constant: float, alpha: float = 0.0) -> torch.Tensor:
    r""" Compute similarity_map between two tensors using Dice-like equation.

    Args:
        map_x: Tensor with map to be compared
        map_y: Tensor with map to be compared
        constant: Used for numerical stability
        alpha: Masking coefficient. Substracts - `alpha` * map_x * map_y from denominator and nominator
    """
    return (2.0 * map_x * map_y - alpha * map_x * map_y + constant) / \
           (map_x ** 2 + map_y ** 2 - alpha * map_x * map_y + constant)


def gradient_map(x: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
    r""" Compute gradient map for a given tensor and stack of kernels.

    Args:
        x: Tensor with shape (N, C, H, W).
        kernels: Stack of tensors for gradient computation with shape (k_N, k_H, k_W)
    Returns:
        Gradients of x per-channel with shape (N, C, H, W)
    """
    padding = kernels.size(-1) // 2
    grads = torch.nn.functional.conv2d(x, kernels.to(x), padding=padding)

    return torch.sqrt(torch.sum(grads ** 2, dim=-3, keepdim=True))


def pow_for_complex(base: torch.Tensor, exp: Union[int, float]) -> torch.Tensor:
    r""" Takes the power of each element in a 4D tensor with negative values or 5D tensor with complex values.
    Complex numbers are represented by modulus and argument: r * \exp(i * \phi).

    It will likely to be redundant with introduction of torch.ComplexTensor.

    Args:
        base: Tensor with shape (N, C, H, W) or (N, C, H, W, 2).
        exp: Exponent
    Returns:
        Complex tensor with shape (N, C, H, W, 2).
    """
    if base.dim() == 4:
        x_complex_r = base.abs()
        x_complex_phi = torch.atan2(torch.zeros_like(base), base)
    elif base.dim() == 5 and base.size(-1) == 2:
        x_complex_r = base.pow(2).sum(dim=-1).sqrt()
        x_complex_phi = torch.atan2(base[..., 1], base[..., 0])
    else:
        raise ValueError(f'Expected real or complex tensor, got {base.size()}')

    x_complex_pow_r = x_complex_r ** exp
    x_complex_pow_phi = x_complex_phi * exp
    x_real_pow = x_complex_pow_r * torch.cos(x_complex_pow_phi)
    x_imag_pow = x_complex_pow_r * torch.sin(x_complex_pow_phi)
    return torch.stack((x_real_pow, x_imag_pow), dim=-1)


def crop_patches(x: torch.Tensor, size=64, stride=32) -> torch.Tensor:
    r"""Crop tensor with images into small patches
    Args:
        x: Tensor with shape (N, C, H, W), expected to be images-like entities
        size: Size of a square patch
        stride: Step between patches
    """
    assert (x.shape[2] >= size) and (x.shape[3] >= size), \
        f"Images must be bigger than patch size. Got ({x.shape[2], x.shape[3]}) and ({size}, {size})"
    channels = x.shape[1]
    patches = x.unfold(1, channels, channels).unfold(2, size, stride).unfold(3, size, stride)
    patches = patches.reshape(-1, channels, size, size)
    return patches
