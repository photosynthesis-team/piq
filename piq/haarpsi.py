"""PyTorch implementation of Haar Wavelet-Based Perceptual Similarity Index (HaarPSI)

Reference:
    [1] R. Reisenhofer, S. Bosse, G. Kutyniok & T. Wiegand (2017)
        A Haar Wavelet-Based Perceptual Similarity Index for Image Quality Assessment
        http://www.math.uni-bremen.de/cda/HaarPSI/publications/HaarPSI_preprint_v4.pdf
    [2] Code from authors on MATLAB and Python
        https://github.com/rgcda/haarpsi
"""
import torch
from typing import Optional, Union

import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from piq.utils import _adjust_dimensions, _validate_input
from piq.functional import similarity_map, rgb2yiq, haar_filter


def haarpsi(x: torch.Tensor, y: torch.Tensor, reduction: Optional[str] = 'mean',
            data_range: Union[int, float] = 1., scales: int = 3, subsample: bool = True,
            C: float = 30.0, alpha: float = 4.2) -> torch.Tensor:
    r"""Compute Haar Wavelet-Based Perceptual Similarity
    Input can by greyscale tensor of colour image with RGB channels order.
    Args:
        x: Tensor of shape :math:`(N, C, H, W)` holding an distorted image.
        y: Tensor of shape :math:`(N, C, H, W)` holding an target image
        reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.
        data_range: The difference between the maximum and minimum of the pixel value,
            i.e., if for image x it holds min(x) = 0 and max(x) = 1, then data_range = 1.
            The pixel value interval of both input and output should remain the same.
        scales: Number of Haar wavelets used for image decomposition.
        subsample: Flag to apply average pooling before HaarPSI computation. See [1] for details.
        C: Constant from the paper. See [1] for details
        alpha: Exponent used for similarity maps weightning. See [1] for details

    Returns:
        HaarPSI : Wavelet-Based Perceptual Similarity between two tensors
    
    Reference:
        [1] R. Reisenhofer, S. Bosse, G. Kutyniok & T. Wiegand (2017)
            'A Haar Wavelet-Based Perceptual Similarity Index for Image Quality Assessment'
            http://www.math.uni-bremen.de/cda/HaarPSI/publications/HaarPSI_preprint_v4.pdf
        [2] Code from authors on MATLAB and Python
            https://github.com/rgcda/haarpsi
    """

    _validate_input(input_tensors=(x, y), allow_5d=False, scale_weights=None)
    x, y = _adjust_dimensions(input_tensors=(x, y))

    # Scale images to [0, 255] range as in the paper
    x = x * 255.0 / float(data_range)
    y = y * 255.0 / float(data_range)

    # Downscale input to simulates the typical distance between an image and its viewer.
    if subsample:
        up_pad = 0
        down_pad = max(x.shape[2] % 2, x.shape[3] % 2)
        pad_to_use = [up_pad, down_pad, up_pad, down_pad]
        x = F.pad(x, pad=pad_to_use)
        y = F.pad(y, pad=pad_to_use)

        x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
        y = F.avg_pool2d(y, kernel_size=2, stride=2, padding=0)

    num_channels = x.size(1)
    # Convert RGB to YIQ color space https://en.wikipedia.org/wiki/YIQ
    if num_channels == 3:
        x = rgb2yiq(x)
        y = rgb2yiq(y)

        x_lum = x[:, : 1]
        y_lum = y[:, : 1]
        x_iq = x[:, 1:]
        y_iq = y[:, 1:]
    else:
        x_lum = x
        y_lum = y

    # Haar wavelet decomposition
    coefficients_x, coefficients_y = [], []
    for scale in range(scales):
        kernel_size = 2 ** (scale + 1)
        kernels = torch.stack([haar_filter(kernel_size), haar_filter(kernel_size).transpose(-1, -2)])
    
        # Assymetrical padding due to even kernel size. Matches MATLAB conv2(A, B, 'same')
        upper_pad = kernel_size // 2 - 1
        bottom_pad = kernel_size // 2
        pad_to_use = [upper_pad, bottom_pad, upper_pad, bottom_pad]
        coeff_x = torch.nn.functional.conv2d(F.pad(x_lum, pad=pad_to_use, mode='constant'), kernels.to(x))
        coeff_y = torch.nn.functional.conv2d(F.pad(y_lum, pad=pad_to_use, mode='constant'), kernels.to(y))
    
        coefficients_x.append(coeff_x)
        coefficients_y.append(coeff_y)

    # Shape B x 6 x H x W
    coefficients_x = torch.cat(coefficients_x, dim=1)
    coefficients_y = torch.cat(coefficients_y, dim=1)

    # Low-frequency coefficients used as weights
    # Shape [B x 2 x H x W]
    weights = torch.max(torch.abs(coefficients_x[:, 4:]), torch.abs(coefficients_y[:, 4:]))
    
    # High-frequency coefficients used for similarity computation in 2 orientations (horizontal and vertical)
    sim_map = []
    for orientation in range(2):
        magnitude_x = torch.abs(coefficients_x[:, (orientation, orientation + 2)])
        magnitude_y = torch.abs(coefficients_y[:, (orientation, orientation + 2)])
        sim_map.append(similarity_map(magnitude_x, magnitude_y, constant=C).sum(dim=1, keepdims=True) / 2)

    if num_channels == 3:
        pad_to_use = [0, 1, 0, 1]
        x_iq = F.pad(x_iq, pad=pad_to_use)
        y_iq = F.pad(y_iq, pad=pad_to_use)
        coefficients_x_iq =  torch.abs(F.avg_pool2d(x_iq, kernel_size=2, stride=1, padding=0))
        coefficients_y_iq =  torch.abs(F.avg_pool2d(y_iq, kernel_size=2, stride=1, padding=0))
    
        # Compute weights and simmilarity
        weights = torch.cat([weights, weights.mean(dim=1, keepdims=True)], dim=1)
        sim_map.append(
            similarity_map(coefficients_x_iq, coefficients_y_iq, constant=C).sum(dim=1, keepdims=True) / 2)

    sim_map = torch.cat(sim_map, dim=1)
    
    # Calculate the final score
    score = torch.sum((sim_map * alpha).sigmoid() * weights, dim=[1, 2, 3]) /\
             torch.sum(weights, dim=[1, 2, 3])
    # Logit of score
    score = (torch.log(score / (1 - score)) / alpha) ** 2

    if reduction == 'none':
        return score

    return {'mean': score.mean,
            'sum': score.sum
            }[reduction](dim=0)
