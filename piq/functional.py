from typing import Tuple, Union
import torch


def ifftshift(x: torch.Tensor):
    r""" Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    shift = [-(ax // 2) for ax in x.size()]
    return torch.roll(x, shift, tuple(range(len(shift))))


def get_meshgrid(size: Tuple[int, int]) -> torch.Tensor:
    r"""
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


# Gradinet operator kernels
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


# Color space conversion
def rgb2lmn(x: torch.Tensor) -> torch.Tensor:
    r"""
    Convert a batch of RGB images to a batch of LMN images

    Args:
        x: Batch of 4D (N x 3 x H x W) images in RGB colour space.

    Returns:
        Batch of 4D (N x 3 x H x W) images in LMN colour space.
    """
    weights_RGB_to_LMN = torch.tensor([[0.06, 0.63, 0.27],
                                       [0.30, 0.04, -0.35],
                                       [0.34, -0.6, 0.17]]).t().to(x)
    x_lmn = torch.matmul(x.permute(0, 2, 3, 1), weights_RGB_to_LMN).permute(0, 3, 1, 2)
    return x_lmn


def rgb2xyz(x: torch.Tensor) -> torch.Tensor:
    r"""
    Convert a batch of RGB images to a batch of XYZ images

    Args:
        x: Batch of 4D (N x 3 x H x W) images in RGB colour space.

    Returns:
        Batch of 4D (N x 3 x H x W) images in XYZ colour space.
    """
    mask_below = (x <= 0.04045).to(x)
    mask_above = (x > 0.04045).to(x)

    tmp = x / 12.92 * mask_below + torch.pow((x + 0.055) / 1.055, 2.4) * mask_above

    weights_RGB_to_XYZ = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                                       [0.2126729, 0.7151522, 0.0721750],
                                       [0.0193339, 0.1191920, 0.9503041]]).to(x)

    x_xyz = torch.matmul(tmp.permute(0, 2, 3, 1), weights_RGB_to_XYZ.t()).permute(0, 3, 1, 2)
    return x_xyz


def xyz2lab(x: torch.Tensor, illuminant='D50', observer='2') -> torch.Tensor:
    r"""
    Convert a batch of XYZ images to a batch of LAB images

    Args:
        x: Batch of 4D (N x 3 x H x W) images in XYZ colour space.
        illuminant: {“A”, “D50”, “D55”, “D65”, “D75”, “E”}, optional. The name of the illuminant.
        observer: {“2”, “10”}, optional. The aperture angle of the observer.
    Returns:
        Batch of 4D (N x 3 x H x W) images in LAB colour space.
    """
    epsilon = 0.008856
    kappa = 903.3
    illuminants = \
        {"A": {'2': (1.098466069456375, 1, 0.3558228003436005),
               '10': (1.111420406956693, 1, 0.3519978321919493)},
         "D50": {'2': (0.9642119944211994, 1, 0.8251882845188288),
                 '10': (0.9672062750333777, 1, 0.8142801513128616)},
         "D55": {'2': (0.956797052643698, 1, 0.9214805860173273),
                 '10': (0.9579665682254781, 1, 0.9092525159847462)},
         "D65": {'2': (0.95047, 1., 1.08883),  # This was: `lab_ref_white`
                 '10': (0.94809667673716, 1, 1.0730513595166162)},
         "D75": {'2': (0.9497220898840717, 1, 1.226393520724154),
                 '10': (0.9441713925645873, 1, 1.2064272211720228)},
         "E": {'2': (1.0, 1.0, 1.0),
               '10': (1.0, 1.0, 1.0)}}

    illuminants_to_use = torch.tensor(illuminants[illuminant][observer]).to(x).view(1, 3, 1, 1)

    tmp = x / illuminants_to_use

    mask_below = tmp <= epsilon
    mask_above = tmp > epsilon
    tmp = torch.pow(tmp, 1. / 3.) * mask_above + (kappa * tmp + 16.) / 116. * mask_below

    weights_XYZ_to_LAB = torch.tensor([[0, 116., 0],
                                       [500., -500., 0],
                                       [0, 200., -200.]]).to(x)
    bias_XYZ_to_LAB = torch.tensor([-16., 0., 0.]).to(x).view(1, 3, 1, 1)

    x_lab = torch.matmul(tmp.permute(0, 2, 3, 1), weights_XYZ_to_LAB.t()).permute(0, 3, 1, 2) + bias_XYZ_to_LAB
    return x_lab


def rgb2lab(x: torch.Tensor, data_range: Union[int, float] = 255) -> torch.Tensor:
    r"""
    Convert a batch of RGB images to a batch of LAB images

    Args:
        x: Batch of 4D (N x 3 x H x W) images in RGB colour space.
        data_range: dynamic range of the input image.
    Returns:
        Batch of 4D (N x 3 x H x W) images in LAB colour space.
    """
    return xyz2lab(rgb2xyz(x / float(data_range)))


def rgb2yiq(x: torch.Tensor) -> torch.Tensor:
    r"""
    Convert a batch of RGB images to a batch of YIQ images

    Args:
        x: Batch of 4D (N x 3 x H x W) images in RGB colour space.

    Returns:
        Batch of 4D (N x 3 x H x W) images in YIQ colour space.
    """
    yiq_weights = torch.tensor([
        [0.299, 0.587, 0.114],
        [0.5959, -0.2746, -0.3213],
        [0.2115, -0.5227, 0.3112]]).t().to(x)
    x_yiq = torch.matmul(x.permute(0, 2, 3, 1), yiq_weights).permute(0, 3, 1, 2)
    return x_yiq
