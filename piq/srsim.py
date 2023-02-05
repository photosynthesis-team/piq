r"""Implementation of Spectral Residual based Similarity
Code is based on MATLAB version for computations in pixel domain
https://github.com/Netflix/vmaf/blob/master/matlab/strred/SR_SIM.m
References:
    https://sse.tongji.edu.cn/linzhang/ICIP12/ICIP-SR-SIM.pdf
"""
import functools
from typing import Union

import torch
import torch.nn.functional as F

from torch.nn.modules.loss import _Loss

from piq.utils import _validate_input, _parse_version, _reduce
from piq.functional import similarity_map, gradient_map, scharr_filter, gaussian_filter, rgb2yiq, imresize


def srsim(x: torch.Tensor, y: torch.Tensor, reduction: str = 'mean',
          data_range: Union[int, float] = 1.0, chromatic: bool = False,
          scale: float = 0.25, kernel_size: int = 3, sigma: float = 3.8,
          gaussian_size: int = 10) -> torch.Tensor:
    r"""Compute Spectral Residual based Similarity for a batch of images.

    Args:
        x: Predicted images. Shape (H, W), (C, H, W) or (N, C, H, W).
        y: Target images. Shape (H, W), (C, H, W) or (N, C, H, W).
        reduction: Reduction over samples in batch: "mean"|"sum"|"none"
        data_range: Value range of input images (usually 1.0 or 255). Default: 1.0
        chromatic: Flag to compute SR-SIMc, which also takes into account chromatic components
        scale: Resizing factor used in saliency map computation
        kernel_size: Kernel size of average blur filter used in saliency map computation
        sigma: Sigma of gaussian filter applied on saliency map
        gaussian_size: Size of gaussian filter applied on saliency map
    Returns:
        SR-SIM: Index of similarity between two images. Usually in [0, 1] interval.
            Can be bigger than 1 for predicted images with higher contrast than the original ones.
    Note:
        This implementation is based on the original MATLAB code.
        https://sse.tongji.edu.cn/linzhang/IQA/SR-SIM/Files/SR_SIM.m

    """
    _validate_input(tensors=[x, y], dim_range=(4, 4), data_range=(0, data_range))

    # Rescale to [0, 255] range, because all constant are calculated for this factor
    x = (x / float(data_range)) * 255
    y = (y / float(data_range)) * 255

    # Averaging image if the size is large enough
    ksize = max(1, round(min(x.size()[-2:]) / 256))
    padding = ksize // 2

    if padding:
        up_pad = (ksize - 1) // 2
        down_pad = padding
        pad_to_use = [up_pad, down_pad, up_pad, down_pad]
        x = F.pad(x, pad=pad_to_use)
        y = F.pad(y, pad=pad_to_use)

    x = F.avg_pool2d(x, ksize)
    y = F.avg_pool2d(y, ksize)

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
        if chromatic:
            raise ValueError('Chromatic component can be computed only for RGB images!')
        x_lum = x
        y_lum = y

    # Compute phase congruency maps
    svrs_x = _spectral_residual_visual_saliency(
        x_lum, scale=scale, kernel_size=kernel_size,
        sigma=sigma, gaussian_size=gaussian_size
    )
    svrs_y = _spectral_residual_visual_saliency(
        y_lum, scale=scale, kernel_size=kernel_size,
        sigma=sigma, gaussian_size=gaussian_size
    )

    # Gradient maps
    sch_filter = scharr_filter(device=x_lum.device, dtype=x_lum.dtype)
    kernels = torch.stack([sch_filter, sch_filter.transpose(-1, -2)])
    grad_map_x = gradient_map(x_lum, kernels)
    grad_map_y = gradient_map(y_lum, kernels)

    # Constants from the paper
    C1, C2, alpha = 0.40, 225, 0.50

    # Compute SR-SIM
    SVRS = similarity_map(svrs_x, svrs_y, C1)
    GM = similarity_map(grad_map_x, grad_map_y, C2)
    svrs_max = torch.where(svrs_x > svrs_y, svrs_x, svrs_y)
    score = SVRS * (GM ** alpha) * svrs_max

    if chromatic:
        # Constants from FSIM paper, use same method for color image
        T3, T4, lmbda = 200, 200, 0.03

        S_I = similarity_map(x_i, y_i, T3)
        S_Q = similarity_map(x_q, y_q, T4)
        score = score * torch.abs(S_I * S_Q) ** lmbda
        # Complex gradients will work in PyTorch 1.6.0
        # score = score * torch.real((S_I * S_Q).to(torch.complex64) ** lmbda)

    eps = torch.finfo(score.dtype).eps
    result = score.sum(dim=[1, 2, 3]) / (svrs_max.sum(dim=[1, 2, 3]) + eps)

    return _reduce(result, reduction)


def _spectral_residual_visual_saliency(x: torch.Tensor, scale: float = 0.25, kernel_size: int = 3,
                                       sigma: float = 3.8, gaussian_size: int = 10) -> torch.Tensor:
    r"""Compute Spectral Residual Visual Saliency
    Credits X. Hou and L. Zhang, CVPR 07, 2007
    Reference:
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.125.5641&rep=rep1&type=pdf

    Args:
        x: Tensor with shape (N, 1, H, W).
        scale: Resizing factor
        kernel_size: Kernel size of average blur filter
        sigma: Sigma of gaussian filter applied on saliency map
        gaussian_size: Size of gaussian filter applied on saliency map
    Returns:
        saliency_map: Tensor with shape BxHxW

    """
    eps = torch.finfo(x.dtype).eps
    for kernel in kernel_size, gaussian_size:
        if x.size(-1) * scale < kernel or x.size(-2) * scale < kernel:
            raise ValueError(f'Kernel size can\'t be greater than actual input size. '
                             f'Input size: {x.size()} x {scale}. Kernel size: {kernel}')

    # Downsize image
    in_img = imresize(x, scale=scale)

    # Fourier transform (use complex format [a,b] instead of a + ib
    # because torch<1.8.0 autograd does not support the latter)
    recommended_torch_version = _parse_version('1.8.0')
    torch_version = _parse_version(torch.__version__)
    if len(torch_version) != 0 and torch_version >= recommended_torch_version:
        imagefft = torch.fft.fft2(in_img)
        log_amplitude = torch.log(imagefft.abs() + eps)
        phase = torch.angle(imagefft)
    else:
        imagefft = torch.rfft(in_img, 2, onesided=False)
        # Compute log of absolute value and angle of fourier transform
        log_amplitude = torch.log(imagefft.pow(2).sum(dim=-1).sqrt() + eps)
        phase = torch.atan2(imagefft[..., 1], imagefft[..., 0] + eps)

    # Compute spectral residual using average filtering
    padding = kernel_size // 2
    if padding:
        up_pad = (kernel_size - 1) // 2
        down_pad = padding
        pad_to_use = [up_pad, down_pad, up_pad, down_pad]
        # replicate padding before average filtering
        spectral_residual = F.pad(log_amplitude, pad=pad_to_use, mode='replicate')
    else:
        spectral_residual = log_amplitude

    spectral_residual = log_amplitude - F.avg_pool2d(spectral_residual, kernel_size=kernel_size, stride=1)
    # Saliency map
    # representation of complex exp(spectral_residual + j * phase)
    compx = torch.stack((
        torch.exp(spectral_residual) * torch.cos(phase),
        torch.exp(spectral_residual) * torch.sin(phase)), -1)

    if len(torch_version) != 0 and torch_version >= recommended_torch_version:
        saliency_map = torch.abs(torch.fft.ifft2(torch.view_as_complex(compx))) ** 2

    else:
        saliency_map = torch.sum(torch.ifft(compx, 2) ** 2, dim=-1)

    # After effect for SR-SIM
    # Apply gaussian blur
    kernel = gaussian_filter(gaussian_size, sigma, device=saliency_map.device, dtype=saliency_map.dtype)
    if gaussian_size % 2 == 0:  # matlab pads upper and lower borders with 0s for even kernels
        kernel = torch.cat((torch.zeros(1, 1, gaussian_size,
                            device=saliency_map.device, dtype=saliency_map.dtype), kernel), 1)
        kernel = torch.cat((torch.zeros(1, gaussian_size + 1, 1,
                            device=saliency_map.device, dtype=saliency_map.dtype), kernel), 2)
        gaussian_size += 1
    kernel = kernel.view(1, 1, gaussian_size, gaussian_size)
    saliency_map = F.conv2d(saliency_map, kernel, padding=(gaussian_size - 1) // 2)

    # normalize between [0, 1]
    min_sal = torch.min(saliency_map[:])
    max_sal = torch.max(saliency_map[:])
    saliency_map = (saliency_map - min_sal) / (max_sal - min_sal + eps)

    # scale to original size
    saliency_map = imresize(saliency_map, sizes=x.size()[-2:])
    return saliency_map


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
        chromatic: Flag to compute SRSIMc, which also takes into account chromatic components
        scale: Resizing factor used in saliency map computation
        kernel_size: Kernel size of average blur filter used in saliency map computation
        sigma: Sigma of gaussian filter applied on saliency map
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

    def __init__(self, reduction: str = 'mean', data_range: Union[int, float] = 1., chromatic: bool = False,
                 scale: float = 0.25, kernel_size: int = 3, sigma: float = 3.8,
                 gaussian_size: int = 10) -> None:
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
            sigma=sigma,
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
        # All checks are done inside srsim function
        score = self.srsim(prediction, target)

        # Make sure value to be in [0, 1] range and convert to loss
        return 1 - torch.clamp(score, 0, 1)
