r"""Implemetation of Spectral Residual based Similarity
Code is based on MATLAB version for computations in pixel domain
https://github.com/Netflix/vmaf/blob/master/matlab/strred/SR_SIM.m
References:
    https://sse.tongji.edu.cn/linzhang/ICIP12/ICIP-SR-SIM.pdf
"""
import math
import functools
from typing import Union, Tuple

import torch
import torch.fft
import torch.nn.functional as F

from torch.nn.modules.loss import _Loss

from piq.utils import _adjust_dimensions, _validate_input
from piq.functional import ifftshift, get_meshgrid, similarity_map, gradient_map, \
                            scharr_filter, gaussian_filter, rgb2yiq
from sklearn.metrics import mean_absolute_error
import cv2
import numpy as np
from scipy import ndimage, signal
from numpy.fft import fft2, ifft2

def srsim(x: torch.Tensor, y: torch.Tensor, reduction: str = 'mean',
         data_range: Union[int, float] = 1.0, chromatic: bool = False,
         scale: float = 0.25, kernel_size: int = 3, gaussian_sigma: float = 3.8, 
         gaussian_size: int = 9) -> torch.Tensor:
    r"""Compute Spectral Residual based Similarity for a batch of images.

    Args:
        x: Predicted images. Shape (H, W), (C, H, W) or (N, C, H, W).
        y: Target images. Shape (H, W), (C, H, W) or (N, C, H, W).
        reduction: Reduction over samples in batch: "mean"|"sum"|"none"
        data_range: Value range of input images (usually 1.0 or 255). Default: 1.0
        chromatic: Flag to compute SR-SIMc, which also takes into account chromatic components
        scale: Resizing factor used in saliency map computation
        kernel_size: Kernel size of average blur filter used in saliency map computation
        gaussian_sigma: Sigma of gaussian filter applied on saliency map
        gaussian_size: Size of gaussian filter applied on saliency map
    Returns:
        SR-SIM: Index of similarity betwen two images. Usually in [0, 1] interval.
            Can be bigger than 1 for predicted images with higher contrast than the original ones.
    Note:
        This implementation is based on the original MATLAB code.
        https://sse.tongji.edu.cn/linzhang/IQA/SR-SIM/Files/SR_SIM.m

    """

    _validate_input(input_tensors=(x, y), allow_5d=False)
    x, y = _adjust_dimensions(input_tensors=(x, y))
    
    # Rescale to [0, 255] range, because all constant are calculated for this factor
    x = (x / float(data_range)) * 255
    y = (y / float(data_range)) * 255
    
    # Apply average pooling
    ksize = max(1, round(min(x.shape[-2:]) / 256))
    x = torch.nn.functional.avg_pool2d(x, ksize)
    y = torch.nn.functional.avg_pool2d(y, ksize)

    num_channels = x.size(1)

    # Convert RGB to YIQ color space https://en.wikipedia.org/wiki/YIQ
    if num_channels == 3:
        x_yiq = rgb2yiq(x)
        y_yiq = rgb2yiq(y)

        x_lum = x_yiq[:, : 1]
        y_lum = y_yiq[:, : 1]
        
        x_i = x_yiq[:, 1:2]
        y_i = y_yiq[:, 1:2]
        x_q = x_yiq[:, 2:]
        y_q = y_yiq[:, 2:]

    else:
        x_lum = x
        y_lum = y

    # Compute phase congruency maps
    print("Size x luminance chan", (x_lum.size()))
    svrs_x = _spectral_residual_visual_saliency(
        x_lum, scale=scale, kernel_size=kernel_size,
        gaussian_sigma=gaussian_sigma, gaussian_size=gaussian_size
    )
    svrs_y = _spectral_residual_visual_saliency(
        y_lum, scale=scale, kernel_size=kernel_size,
        gaussian_sigma=gaussian_sigma, gaussian_size=gaussian_size
    )
    print("Size x svrx result", svrs_x.size())

    # Gradient maps
    kernels = torch.stack([scharr_filter(), scharr_filter().transpose(-1, -2)])
    grad_map_x = gradient_map(x_lum, kernels)
    grad_map_y = gradient_map(y_lum, kernels)

    # Constants from the paper
    C1, C2, alpha = 0.40, 225, 0.50

    # Compute SR-SIM
    SVRS = similarity_map(svrs_x, svrs_y, C1)
    GM = similarity_map(grad_map_x, grad_map_y, C2)
    svrs_max = torch.where(svrs_x > svrs_y, svrs_x, svrs_y)
    score = GM * SVRS * svrs_max

    if chromatic:
        assert prediction.size(1) == 3, "Chromatic component can be computed only for RGB images!"

        # Constants from FSIM paper, use same method for color image
        T3, T4, lmbda = 200, 200, 0.03

        S_I = similarity_map(x_i, y_i, T3)
        S_Q = similarity_map(x_q, y_q, T4)
        score = score * torch.abs(S_I * S_Q) ** lmbda
        # Complex gradients will work in PyTorch 1.6.0
        # score = score * torch.real((S_I * S_Q).to(torch.complex64) ** lmbda)

    result = score.sum(dim=[1, 2, 3]) / svrs_max.sum(dim=[1, 2, 3])
    
    if reduction == 'none':
        return result

    return {'mean': result.mean,
            'sum': result.sum
            }[reduction](dim=0)


def _spectral_residual_visual_saliency(x: torch.Tensor, scale: float = 0.25, kernel_size: int = 3,
                      gaussian_sigma: float = 3.8, gaussian_size: int = 9) -> torch.Tensor:
    r"""Compute Spectral Residual Visual Saliency
    Credits X. Hou and L. Zhang, CVPR 07, 2007
    Reference:
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.125.5641&rep=rep1&type=pdf

    Args:
        x: Tensor with shape (N, 1, H, W).
        scale: Resizing factor
        kernel_size: Kernel size of average blur filter
        gaussian_sigma: Sigma of gaussian filter applied on saliency map
        gaussian_size: Size of gaussian filter applied on saliency map
    Returns:
        saliency_map: Tensor with shape BxHxW

    """
    # Downsize image
    in_img = F.interpolate(x,
                            scale_factor=scale,
                            mode='bicubic', # bicubic is closer to matlab 'imresize' implementation
                            align_corners=False)

    # Fourier transform (complex number)
    imagefft = torch.view_as_complex(torch.rfft(in_img, 2, onesided=False))

    # Compute log amplitude and angle of fourier transform
    log_amplitude = torch.log(imagefft.abs())
    phase = imagefft.angle()

    # Compute spectral residual using average filtering
    assert(kernel_size/2==0 and gaussian_size/2==0, 'Kernel size must be divisible by 2')
    average_filter = torch.nn.AvgPool2d(kernel_size, stride=1, padding=(kernel_size-1)//2) # avg pool equivalent to filter if stride == 1
    spectral_residual = log_amplitude - average_filter(log_amplitude)

    # Saliency map
    saliency_map_base = torch.abs(torch.fft.ifft(torch.exp(spectral_residual + 1j * phase))) ** 2

    # After effect for SR-SIM
    # Apply gaussian blur
    kernel = gaussian_filter(gaussian_size, gaussian_sigma).view(1, 1, gaussian_size, gaussian_size).to(saliency_map_base)
    saliency_map = F.conv2d(saliency_map_base, kernel, padding=(gaussian_size-1)//2)
    # normalize between [0, 1]
    saliency_map -= torch.min(saliency_map)
    saliency_map /= torch.max(saliency_map)

    # scale to original size
    return F.interpolate(saliency_map, scale_factor=(1/scale, 1/scale), mode='bicubic', align_corners=False)


class SRSIMLoss(_Loss):
    r"""Creates a criterion that measures the SR-SIM or SR-SIMc for input :math:`x` and target :math:`y`.

    In order to be considered as a loss, value `1 - clip(SR-SIM, min=0, max=1)` is returned. If you need SR-SIM value,
    use function `srsim` instead.

    Args:
        reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        data_range: The difference between the maximum and minimum of the pixel value,
            i.e., if for image x it holds min(x) = 0 and max(x) = 1, then data_range = 1.
            The pixel value interval of both input and output should remain the same.
        chromatic: Flag to compute FSIMc, which also takes into account chromatic components
        scale: Resizing factor used in saliency map computation
        kernel_size: Kernel size of average blur filter used in saliency map computation
        gaussian_sigma: Sigma of gaussian filter applied on saliency map
        gaussian_size: Size of gaussian filter applied on saliency map

    Shape:
        - Input: Required to be 2D (H, W), 3D (C, H, W) or 4D (N, C, H, W). RGB channel order for colour images.
        - Target: Required to be 2D (H, W), 3D (C, H, W) or 4D (N, C, H, W). RGB channel order for colour images.

    Examples::

        >>> loss = SRSIMLoss()
        >>> prediction = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> target = torch.rand(3, 3, 256, 256)
        >>> output = loss(prediction, target)
        >>> output.backward()

    References:
        https://sse.tongji.edu.cn/linzhang/ICIP12/ICIP-SR-SIM.pdf
        """
    def __init__(self, reduction: str = 'mean', data_range: Union[int, float] = 1., chromatic: bool = True,
                 scale: float = 0.25, kernel_size: int = 3, gaussian_sigma: float = 3.8, 
                 gaussian_size: int = 9) -> None:
        super().__init__()
        self.data_range = data_range
        self.reduction = reduction

        # Save function with predefined parameters, rather than parameters themself
        self.srsim = functools.partial(
            srsim,
            reduction=reduction,
            data_range=data_range,
            chromatic=chromatic,
            scale=scale,
            kernel_size=kernel_size,
            gaussian_sigma=gaussian_sigma,
            gaussian_size=gaussian_size
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Computation of SR-SIM as a loss function.
        Args:
            prediction: Tensor of prediction of the network.
            target: Reference tensor.
        Returns:
            Value of SR-SIM loss to be minimized. 0 <= SR-SIM <= 1.
        """
        # All checks are done inside fsim function
        score = self.srsim(prediction, target)

        # Make sure value to be in [0, 1] range and convert to loss
        return 1 - torch.clamp(score, 0, 1)


if __name__ == "__main__":
    from skimage.io import imread

    # Greyscale images
    goldhill = torch.tensor(imread('tests/assets/goldhill.gif'))
    goldhill_jpeg = torch.tensor(imread('tests/assets/goldhill_jpeg.gif'))

    score = srsim(goldhill_jpeg, goldhill, data_range=255, chromatic=False, reduction='none')
    print(score)
