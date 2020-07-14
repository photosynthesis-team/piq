r"""Implemetation of Visual Saliency-induced Index
Code is based on the MATLAB version for computations in pixel domain
https://sse.tongji.edu.cn/linzhang/IQA/VSI/VSI.htm

References:
    https://ieeexplore.ieee.org/document/6873260
"""
from typing import Union, Tuple
import torch
from torch.nn.modules.loss import _Loss
from torch.nn.functional import avg_pool2d, interpolate, pad
from piq.functional import ifftshift, gradient_map, scharr_filter, rgb2lmn, rgb2lab, similarity_map, get_meshgrid
from piq.utils import _validate_input, _adjust_dimensions
import functools
import warnings


def vsi(prediction: torch.Tensor, target: torch.Tensor, reduction: str = 'mean', data_range: Union[int, float] = 1.,
        c1: float = 1.27, c2: float = 386., c3: float = 130., alpha: float = 0.4, beta: float = 0.02,
        omega_0: float = 0.021, sigma_f: float = 1.34, sigma_d: float = 145., sigma_c: float = 0.001) -> torch.Tensor:
    r"""Compute Visual Saliency-induced Index for a batch of images.

        Both inputs are supposed to have RGB order in accordance with the original approach.
        Nevertheless, the method supports greyscale images, which they are converted to RGB by copying the grey
        channel 3 times.

        Args:
            prediction: Batch of predicted images with shape (batch_size x channels x H x W)
            target: Batch of target images with shape  (batch_size x channels x H x W)
            reduction: Reduction over samples in batch: "mean"|"sum"|"none"
            data_range: Value range of input images (usually 1.0 or 255). Default: 1.0
            c1: coefficient to calculate saliency component of VSI
            c2: coefficient to calculate gradient component of VSI
            c3: coefficient to calculate color component of VSI
            alpha: power for gradient component of VSI
            beta: power for color component of VSI
            omega_0: coefficient to get log Gabor filter at SDSP
            sigma_f: coefficient to get log Gabor filter at SDSP
            sigma_d: coefficient to get SDSP
            sigma_c: coefficient to get SDSP

        Returns:
            VSI: Index of similarity between two images. Usually in [0, 1] interval.

        Shape:
                - Input: Required to be 2D (H,W), 3D (C,H,W), 4D (N,C,H,W), channels first.
                - Target: Required to be 2D (H,W), 3D (C,H,W), 4D (N,C,H,W), channels first.
        Note:
            The original method supports only RGB image.
            See https://ieeexplore.ieee.org/document/6873260 for details.
        """
    _validate_input(input_tensors=(prediction, target), allow_5d=False)
    prediction, target = _adjust_dimensions(input_tensors=(prediction, target))
    if prediction.size(1) == 1:
        prediction = prediction.repeat(1, 3, 1, 1)
        target = target.repeat(1, 3, 1, 1)
        warnings.warn('The original VSI supports only RGB images. The input images were converted to RGB by copying '
                      'the grey channel 3 times.')

    # Scale to [0, 255] range to match scale of constant
    prediction = prediction * 255. / float(data_range)
    target = target * 255. / float(data_range)

    vs_prediction = sdsp(prediction, data_range=255, omega_0=omega_0,
                         sigma_f=sigma_f, sigma_d=sigma_d, sigma_c=sigma_c)
    vs_target = sdsp(target, data_range=255, omega_0=omega_0, sigma_f=sigma_f,
                     sigma_d=sigma_d, sigma_c=sigma_c)

    # Convert to LMN colour space
    prediction_lmn = rgb2lmn(prediction)
    target_lmn = rgb2lmn(target)

    # Averaging image if the size is large enough
    kernel_size = max(1, round(min(vs_prediction.size()[-2:]) / 256))
    padding = kernel_size // 2

    if padding:
        vs_prediction = pad(vs_prediction, pad=[1, 0, 1, 0], mode='replicate')
        vs_target = pad(vs_target, pad=[1, 0, 1, 0], mode='replicate')
        prediction_lmn = pad(prediction_lmn, pad=[1, 0, 1, 0], mode='replicate')
        target_lmn = pad(target_lmn, pad=[1, 0, 1, 0], mode='replicate')

    vs_prediction = avg_pool2d(vs_prediction, kernel_size=kernel_size)
    vs_target = avg_pool2d(vs_target, kernel_size=kernel_size)

    prediction_lmn = avg_pool2d(prediction_lmn, kernel_size=kernel_size)
    target_lmn = avg_pool2d(target_lmn, kernel_size=kernel_size)

    # Calculate gradient map
    kernels = torch.stack([scharr_filter(), scharr_filter().transpose(1, 2)]).to(prediction_lmn)
    gm_prediction = gradient_map(prediction_lmn[:, :1], kernels)
    gm_target = gradient_map(target_lmn[:, :1], kernels)

    # Calculate all similarity maps
    s_vs = similarity_map(vs_prediction, vs_target, c1)
    s_gm = similarity_map(gm_prediction, gm_target, c2)
    s_m = similarity_map(prediction_lmn[:, 1:2], target_lmn[:, 1:2], c3)
    s_n = similarity_map(prediction_lmn[:, 2:], target_lmn[:, 2:], c3)
    s_c = s_m * s_n

    s_c_complex = [s_c.abs(), torch.atan2(torch.zeros_like(s_c), s_c)]
    s_c_complex_pow = [s_c_complex[0] ** beta, s_c_complex[1] * beta]
    s_c_real_pow = s_c_complex_pow[0] * torch.cos(s_c_complex_pow[1])

    s = s_vs * s_gm.pow(alpha) * s_c_real_pow
    vs_max = torch.max(vs_prediction, vs_target)

    eps = torch.finfo(vs_max.dtype).eps
    output = s * vs_max
    output = ((output.sum(dim=(-1, -2)) + eps) / (vs_max.sum(dim=(-1, -2)) + eps)).squeeze(-1)
    if reduction == 'none':
        return output
    return {'mean': torch.mean,
            'sum': torch.sum
            }[reduction](output, dim=0)


class VSILoss(_Loss):
    def __init__(self, reduction: str = 'mean', c1: float = 1.27, c2: float = 386., c3: float = 130.,
                 alpha: float = 0.4, beta: float = 0.02, data_range: Union[int, float] = 1.,
                 omega_0: float = 0.021, sigma_f: float = 1.34, sigma_d: float = 145., sigma_c: float = 0.001) -> None:
        r"""Creates a criterion that measures Visual Saliency-induced Index error between
            each element in the input and target.

            The sum operation still operates over all the elements, and divides by :math:`n`.

            The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

            Args:
                reduction: Reduction over samples in batch: "mean"|"sum"|"none"
                data_range: Value range of input images (usually 1.0 or 255). Default: 1.0
                c1: coefficient to calculate saliency component of VSI
                c2: coefficient to calculate gradient component of VSI
                c3: coefficient to calculate color component of VSI
                alpha: power for gradient component of VSI
                beta: power for color component of VSI
                omega_0: coefficient to get log Gabor filter at SDSP
                sigma_f: coefficient to get log Gabor filter at SDSP
                sigma_d: coefficient to get SDSP
                sigma_c: coefficient to get SDSP

            Shape:
                - Input: Required to be 2D (H,W), 3D (C,H,W), 4D (N,C,H,W), channels first.
                - Target: Required to be 2D (H,W), 3D (C,H,W), 4D (N,C,H,W), channels first.

                Both inputs are supposed to have RGB order in accordance with the original approach.
                Nevertheless, the method supports greyscale images, which they are converted to RGB
                by copying the grey channel 3 times.

            Examples::

                >>> loss = VSILoss()
                >>> prediction = torch.rand(3, 3, 256, 256, requires_grad=True)
                >>> target = torch.rand(3, 3, 256, 256)
                >>> output = loss(prediction, target)
                >>> output.backward()

            References:
                .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
                   (2004). Image quality assessment: From error visibility to
                   structural similarity. IEEE Transactions on Image Processing,
                   13, 600-612.
                   https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
                   :DOI:`10.1109/TIP.2003.819861`
            """
        super().__init__()
        self.reduction = reduction
        self.data_range = data_range

        self.vsi = functools.partial(
            vsi, c1=c1, c2=c2, c3=c3, alpha=alpha, beta=beta, omega_0=omega_0,
            sigma_f=sigma_f, sigma_d=sigma_d, sigma_c=sigma_c, data_range=data_range, reduction=reduction)

    def forward(self, prediction, target):
        r"""Computation of VSI as a loss function.

            Args:
                prediction: Tensor of prediction of the network.
                target: Reference tensor.

            Returns:
                Value of VSI loss to be minimized. 0 <= VSI loss <= 1.

            Note:
                Both inputs are supposed to have RGB order in accordance with the original approach.
                Nevertheless, the method supports greyscale images, which they are converted to RGB by copying the grey
                channel 3 times.
        """

        return 1. - self.vsi(prediction=prediction, target=target)


def sdsp(x: torch.Tensor, data_range: Union[int, float] = 255, omega_0: float = 0.021, sigma_f: float = 1.34,
         sigma_d: float = 145., sigma_c: float = 0.001) -> torch.Tensor:
    r"""
    SDSP algorithm for salient region detection from a given image.

    Args :
        x: an  RGB image with dynamic range [0, 1] or [0, 255] for each channel
        data_range: dynamic range of the image
        omega_0: coefficient for log Gabor filter
        sigma_f: coefficient for log Gabor filter
        sigma_d: coefficient for the central areas, which have a bias towards attention
        sigma_c: coefficient for the warm colors, which have a bias towards attention

    Returns:
        torch.Tensor: Visual saliency map
    """
    x = x * 255. / float(data_range)
    size = x.size()
    size_to_use = (256, 256)
    x = interpolate(input=x, size=size_to_use, mode='bilinear', align_corners=False)

    x_lab = rgb2lab(x, data_range=255)
    x_fft = torch.rfft(x_lab, 2, onesided=False)
    lg = _log_gabor(size_to_use, omega_0, sigma_f).to(x_fft).view(1, 1, *size_to_use, 1)
    x_ifft_real = torch.ifft(x_fft * lg, 2)[..., 0]
    s_f = x_ifft_real.pow(2).sum(dim=1, keepdim=True).sqrt()

    coordinates = torch.stack((get_meshgrid(size_to_use)), dim=0).to(x)
    coordinates = coordinates * size_to_use[0] + 1
    s_d = torch.exp(-torch.sum(coordinates ** 2, dim=0) / sigma_d ** 2).view(1, 1, *size_to_use)

    eps = torch.finfo(x_lab.dtype).eps
    min_x = x_lab.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values
    max_x = x_lab.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values
    normalized = (x_lab - min_x) / (max_x - min_x + eps)

    norm = normalized[:, 1:].pow(2).sum(dim=1, keepdim=True)
    s_c = 1 - torch.exp(-norm / sigma_c ** 2)

    vs_m = s_f * s_d * s_c
    vs_m = interpolate(vs_m, size[-2:], mode='bilinear', align_corners=True)
    min_vs_m = vs_m.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values
    max_vs_m = vs_m.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values
    return (vs_m - min_vs_m) / (max_vs_m - min_vs_m + eps)


def _log_gabor(size: Tuple[int, int], omega_0: float, sigma_f: float) -> torch.Tensor:
    r"""
    Creates log Gabor filter
    Args:
        size: size of the requires log Gabor filter
        omega_0: center frequency of the filter
        sigma_f: bandwidth of the filter

    Returns:
        log Gabor filter
    """
    xx, yy = get_meshgrid(size)

    radius = (xx ** 2 + yy ** 2).sqrt()
    mask = radius <= 0.5

    r = radius * mask
    r = ifftshift(r)
    r[0, 0] = 1

    lg = torch.exp((-(r / omega_0).log().pow(2)) / (2 * sigma_f ** 2))
    lg[0, 0] = 0
    return lg
