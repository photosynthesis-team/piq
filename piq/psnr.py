r""" This module implements Peak Signal-to-Noise Ratio (PSNR) in PyTorch.
"""
import torch
from typing import Union

from piq.utils import _validate_input, _adjust_dimensions


def psnr(x: torch.Tensor, y: torch.Tensor, data_range: Union[int, float] = 1.0,
         reduction: str = 'mean', convert_to_greyscale: bool = False) -> torch.Tensor:
    r"""Compute Peak Signal-to-Noise Ratio for a batch of images.
    Supports both greyscale and color images with RGB channel order.

    Args:
        x: Predicted images. Shape (H, W), (C, H, W) or (N, C, H, W).
        y: Target images. Shape (H, W), (C, H, W) or (N, C, H, W).
        data_range: Value range of input images (usually 1.0 or 255). Default: 1.0
        reduction: Reduction over samples in batch: "mean"|"sum"|"none"
        convert_to_greyscale: Convert RGB image to YCbCr format and computes PSNR
            only on luminance channel if `True`. Compute on all 3 channels otherwise.

    Returns:
        PSNR: Index of similarity betwen two images.

    Description:
        Peak signal-to-noise ratio (PSNR) is an engineering term for the ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation. Because many signals have a very wide dynamic range, PSNR is usually expressed as a logarithmic quantity using the decibel scale. 
        PSNR is most easily defined via the mean squared error $(MSE)$. Given a noise-free m×n monochrome image $I$ and its noisy approximation $K$, $MSE$ is defined as:
        $$
            {\mathit {MSE}}={\frac {1}{m\,n}}\sum _{{i=0}}^{{m-1}}\sum _{{j=0}}^{{n-1}}[I(i,j)-K(i,j)]^{2}
        $$
        The PSNR (in dB) is defined as:
        $$
            {\displaystyle {\begin{aligned}{\mathit {PSNR}}&=10\cdot \log _{10}\left({\frac {{\mathit {MAX}}_{I}^{2}}{\mathit {MSE}}}\right)\\&=20\cdot \log _{10}\left({\frac {{\mathit {MAX}}_{I}}{\sqrt {\mathit {MSE}}}}\right)\\&=20\cdot \log _{10}\left({{\mathit {MAX}}_{I}}\right)-10\cdot \log _{10}\left({\mathit {MSE}}\right)\end{aligned}}} 
        $$
        Here, MAXI is the maximum possible pixel value of the image. When the pixels are represented using 8 bits per sample, this is 255. More generally, when samples are represented using linear PCM with B bits per sample, $MAX_I$ is $2^B−1$.

        For color images with three RGB values per pixel, the definition of PSNR is the same except the MSE is the sum over all squared value differences (now for each color, i.e. three times as many differences as in a monochrome image) divided by image size and by three. Alternately, for color images the image is converted to a different color space and PSNR is reported against each channel of that color space, e.g., YCbCr or HSL

    References:
        https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    _validate_input((x, y), allow_5d=False, data_range=data_range)
    x, y = _adjust_dimensions(input_tensors=(x, y))

    # Constant for numerical stability
    EPS = 1e-8

    x = x / data_range
    y = y / data_range

    if (x.size(1) == 3) and convert_to_greyscale:
        # Convert RGB image to YCbCr and take luminance: Y = 0.299 R + 0.587 G + 0.114 B
        rgb_to_grey = torch.tensor([0.299, 0.587, 0.114]).view(1, -1, 1, 1).to(x)
        x = torch.sum(x * rgb_to_grey, dim=1, keepdim=True)
        y = torch.sum(y * rgb_to_grey, dim=1, keepdim=True)

    mse = torch.mean((x - y) ** 2, dim=[1, 2, 3])
    score: torch.Tensor = - 10 * torch.log10(mse + EPS)

    if reduction == 'none':
        return score

    return {'mean': score.mean,
            'sum': score.sum
            }[reduction](dim=0)
