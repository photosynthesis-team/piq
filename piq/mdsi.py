r"""Implementation of Mean Deviation Similarity Index (MDSI)
Code supports the functionality proposed with the original MATLAB version for computations in pixel domain
https://www.mathworks.com/matlabcentral/fileexchange/59809

References:
    https://arxiv.org/pdf/1608.07433.pdf
"""
import warnings
import functools
from typing import Union

import torch
from torch.nn.modules.loss import _Loss
from torch.nn.functional import pad, avg_pool2d

from piq.functional import rgb2lhm, gradient_map, similarity_map, prewitt_filter, pow_for_complex
from piq.utils import _validate_input, _reduce


def mdsi(x: torch.Tensor, y: torch.Tensor, data_range: Union[int, float] = 1., reduction: str = 'mean',
         c1: float = 140., c2: float = 55., c3: float = 550., combination: str = 'sum', alpha: float = 0.6,
         beta: float = 0.1, gamma: float = 0.2, rho: float = 1., q: float = 0.25, o: float = 0.25):
    r"""Compute Mean Deviation Similarity Index (MDSI) for a batch of images.
    Supports greyscale and colour images with RGB channel order.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        data_range: Maximum value range of images (usually 1.0 or 255).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        c1: coefficient to calculate gradient similarity. Default: 140.
        c2: coefficient to calculate gradient similarity. Default: 55.
        c3: coefficient to calculate chromaticity similarity. Default: 550.
        combination: mode to combine gradient similarity and chromaticity similarity: ``'sum'`` | ``'mult'``.
        alpha: coefficient to combine gradient similarity and chromaticity similarity using summation.
        beta: power to combine gradient similarity with chromaticity similarity using multiplication.
        gamma: to combine gradient similarity and chromaticity similarity using multiplication.
        rho: order of the Minkowski distance
        q: coefficient to adjusts the emphasis of the values in image and MCT
        o: the power pooling applied on the final value of the deviation

    Returns:
        Mean Deviation Similarity Index (MDSI) between 2 tensors.

    References:
        Nafchi, Hossein Ziaei and Shahkolaei, Atena and Hedjam, Rachid and Cheriet, Mohamed (2016).
        Mean deviation similarity index: Efficient and reliable full-reference image quality evaluator.
        IEEE Ieee Access, 4, 5579--5590.
        https://arxiv.org/pdf/1608.07433.pdf,
        DOI:`10.1109/ACCESS.2016.2604042`

    Note:
        The ratio between constants is usually equal :math:`c_3 = 4c_1 = 10c_2`

    Note:
        Both inputs are supposed to have RGB channels order in accordance with the original approach.
        Nevertheless, the method supports greyscale images, which are converted to RGB by copying the grey
        channel 3 times.
    """
    _validate_input([x, y], dim_range=(4, 4), data_range=(0, data_range))

    if x.size(1) == 1:
        x = x.repeat(1, 3, 1, 1)
        y = y.repeat(1, 3, 1, 1)
        warnings.warn('The original MDSI supports only RGB images. The input images were converted to RGB by copying '
                      'the grey channel 3 times.')

    x = x / float(data_range) * 255
    y = y / float(data_range) * 255

    # Averaging image if the size is large enough
    kernel_size = max(1, round(min(x.size()[-2:]) / 256))
    padding = kernel_size // 2

    if padding:
        up_pad = (kernel_size - 1) // 2
        down_pad = padding
        pad_to_use = [up_pad, down_pad, up_pad, down_pad]
        x = pad(x, pad=pad_to_use)
        y = pad(y, pad=pad_to_use)

    x = avg_pool2d(x, kernel_size=kernel_size)
    y = avg_pool2d(y, kernel_size=kernel_size)

    x_lhm = rgb2lhm(x)
    y_lhm = rgb2lhm(y)

    kernels = torch.stack([prewitt_filter(), prewitt_filter().transpose(1, 2)]).to(x)
    gm_x = gradient_map(x_lhm[:, :1], kernels)
    gm_y = gradient_map(y_lhm[:, :1], kernels)
    gm_avg = gradient_map((x_lhm[:, :1] + y_lhm[:, :1]) / 2., kernels)

    gs_x_y = similarity_map(gm_x, gm_y, c1)
    gs_x_average = similarity_map(gm_x, gm_avg, c2)
    gs_y_average = similarity_map(gm_y, gm_avg, c2)

    gs_total = gs_x_y + gs_x_average - gs_y_average

    cs_total = (2 * (x_lhm[:, 1:2] * y_lhm[:, 1:2] +
                     x_lhm[:, 2:] * y_lhm[:, 2:]) + c3) / (x_lhm[:, 1:2] ** 2 +
                                                           y_lhm[:, 1:2] ** 2 +
                                                           x_lhm[:, 2:] ** 2 +
                                                           y_lhm[:, 2:] ** 2 + c3)

    if combination == 'sum':
        gcs = (alpha * gs_total + (1 - alpha) * cs_total)
    elif combination == 'mult':
        gs_total_pow = pow_for_complex(base=gs_total, exp=gamma)
        cs_total_pow = pow_for_complex(base=cs_total, exp=beta)
        gcs = torch.stack((gs_total_pow[..., 0] * cs_total_pow[..., 0],
                           gs_total_pow[..., 1] + cs_total_pow[..., 1]), dim=-1)
    else:
        raise ValueError(f'Expected combination method "sum" or "mult", got {combination}')

    mct_complex = pow_for_complex(base=gcs, exp=q)
    mct_complex = mct_complex.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)  # split to increase precision
    score = (pow_for_complex(base=gcs, exp=q) - mct_complex).pow(2).sum(dim=-1).sqrt()
    score = ((score ** rho).mean(dim=(-1, -2)) ** (o / rho)).squeeze(1)
    return _reduce(score, reduction)


class MDSILoss(_Loss):
    r"""Creates a criterion that measures Mean Deviation Similarity Index (MDSI) error between the prediction :math:`x`
    and target :math:`y`.
    Supports greyscale and colour images with RGB channel order.

    Args:
        data_range: Maximum value range of images (usually 1.0 or 255).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        c1: coefficient to calculate gradient similarity. Default: 140.
        c2: coefficient to calculate gradient similarity. Default: 55.
        c3: coefficient to calculate chromaticity similarity. Default: 550.
        combination: mode to combine gradient similarity and chromaticity similarity: ``'sum'`` | ``'mult'``.
        alpha: coefficient to combine gradient similarity and chromaticity similarity using summation.
        beta: power to combine gradient similarity with chromaticity similarity using multiplication.
        gamma: to combine gradient similarity and chromaticity similarity using multiplication.
        rho: order of the Minkowski distance
        q: coefficient to adjusts the emphasis of the values in image and MCT
        o: the power pooling applied on the final value of the deviation

    Examples:
        >>> loss = MDSILoss(data_range=1.)
        >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> y = torch.rand(3, 3, 256, 256)
        >>> output = loss(x, y)
        >>> output.backward()

    References:
        Nafchi, Hossein Ziaei and Shahkolaei, Atena and Hedjam, Rachid and Cheriet, Mohamed (2016).
        Mean deviation similarity index: Efficient and reliable full-reference image quality evaluator.
        IEEE Ieee Access, 4, 5579--5590.
        https://arxiv.org/pdf/1608.07433.pdf
        DOI:`10.1109/ACCESS.2016.2604042`

    Note:
        The ratio between constants is usually equal :math:`c_3 = 4c_1 = 10c_2`
    """

    def __init__(self, data_range: Union[int, float] = 1., reduction: str = 'mean',
                 c1: float = 140., c2: float = 55., c3: float = 550., alpha: float = 0.6,
                 rho: float = 1., q: float = 0.25, o: float = 0.25, combination: str = 'sum',
                 beta: float = 0.1, gamma: float = 0.2):
        super().__init__()
        self.reduction = reduction
        self.data_range = data_range
        self.mdsi = functools.partial(mdsi, c1=c1, c2=c2, c3=c3, alpha=alpha, rho=rho, q=q, o=o,
                                      combination=combination, beta=beta, gamma=gamma, data_range=self.data_range,
                                      reduction=self.reduction)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""Computation of Mean Deviation Similarity Index (MDSI) as a loss function.

        Both inputs are supposed to have RGB channels order.
        Greyscale images converted to RGB by copying the grey channel 3 times.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Value of MDSI loss to be minimized in [0, 1] range.

        Note:
            Both inputs are supposed to have RGB channels order in accordance with the original approach.
            Nevertheless, the method supports greyscale images, which are converted to RGB by copying the grey
            channel 3 times.
        """
        return self.mdsi(x=x, y=y)
