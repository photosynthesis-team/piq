r"""Implemetation of Mean Deviation Similarity Index (MDSI)
Code supports the functionality proposed with the original MATLAB version for computations in pixel domain
https://www.mathworks.com/matlabcentral/fileexchange/59809

References:
    https://arxiv.org/pdf/1608.07433.pdf
"""
import warnings
import functools
import torch
from torch.nn.modules.loss import _Loss
from torch.nn.functional import pad, avg_pool2d
from typing import Union
from piq.functional import rgb2lhm, gradient_map, similarity_map, prewitt_filter, pow_for_complex
from piq.utils import _validate_input, _adjust_dimensions


def mdsi(prediction: torch.Tensor, target: torch.Tensor, data_range: Union[int, float] = 1., reduction: str = 'mean',
         c1: float = 140., c2: float = 55., c3: float = 550., combination: str = 'sum', alpha: float = 0.6,
         beta: float = 0.1, gamma: float = 0.2, rho: float = 1., q: float = 0.25, o: float = 0.25):
    r"""Compute Mean Deviation Similarity Index (MDSI) for a batch of images.

    Note:
        Both inputs are supposed to have RGB channels order.
        Greyscale images converted to RGB by copying the grey channel 3 times.

    Args:
        prediction: Tensor with shape (H, W), (C, H, W) or (N, C, H, W).
        target:Tensor with shape (H, W), (C, H, W) or (N, C, H, W).
        data_range: Value range of input images (usually 1.0 or 255). Default: 1.0
        reduction: Reduction over samples in batch: "mean"|"sum"|"none"
        c1: coefficient to calculate gradient similarity. Default: 140.
        c2: coefficient to calculate gradient similarity. Default: 55.
        c3: coefficient to calculate chromaticity similarity. Default: 550.
        combination: mode to combine gradient similarity and chromaticity similarity: "sum"|"mult".
        alpha: coefficient to combine gradient similarity and chromaticity similarity using summation.
        beta: power to combine gradient similarity with chromaticity similarity using multiplication.
        gamma: to combine gradient similarity and chromaticity similarity using multiplication.
        rho: order of the Minkowski distance
        q: coefficient to adjusts the emphasis of the values in image and MCT
        o: the power pooling applied on the final value of the deviation

    Returns:
        torch.Tensor: the batch of Mean Deviation Similarity Index (MDSI) score reduced accordingly

    Note:
        The ratio between constants is usually equal c3 = 4c1 = 10c2
    """
    _validate_input(input_tensors=(prediction, target), allow_5d=False, data_range=data_range)
    prediction, target = _adjust_dimensions(input_tensors=(prediction, target))

    if prediction.size(1) == 1:
        prediction = prediction.repeat(1, 3, 1, 1)
        target = target.repeat(1, 3, 1, 1)
        warnings.warn('The original MDSI supports only RGB images. The input images were converted to RGB by copying '
                      'the grey channel 3 times.')

    prediction = prediction / data_range * 255
    target = target / data_range * 255

    # Averaging image if the size is large enough
    kernel_size = max(1, round(min(prediction.size()[-2:]) / 256))
    padding = kernel_size // 2

    if padding:
        up_pad = (kernel_size - 1) // 2
        down_pad = padding
        pad_to_use = [up_pad, down_pad, up_pad, down_pad]
        prediction = pad(prediction, pad=pad_to_use)
        target = pad(target, pad=pad_to_use)

    prediction = avg_pool2d(prediction, kernel_size=kernel_size)
    target = avg_pool2d(target, kernel_size=kernel_size)

    prediction_lhm = rgb2lhm(prediction)
    target_lhm = rgb2lhm(target)

    kernels = torch.stack([prewitt_filter(), prewitt_filter().transpose(1, 2)]).to(prediction)
    gm_prediction = gradient_map(prediction_lhm[:, :1], kernels)
    gm_target = gradient_map(target_lhm[:, :1], kernels)
    gm_avg = gradient_map((prediction_lhm[:, :1] + target_lhm[:, :1]) / 2., kernels)

    gs_prediction_target = similarity_map(gm_prediction, gm_target, c1)
    gs_prediction_average = similarity_map(gm_prediction, gm_avg, c2)
    gs_target_average = similarity_map(gm_target, gm_avg, c2)

    gs_total = gs_prediction_target + gs_prediction_average - gs_target_average

    cs_total = (2 * (prediction_lhm[:, 1:2] * target_lhm[:, 1:2] +
                     prediction_lhm[:, 2:] * target_lhm[:, 2:]) + c3) / (prediction_lhm[:, 1:2] ** 2 +
                                                                         target_lhm[:, 1:2] ** 2 +
                                                                         prediction_lhm[:, 2:] ** 2 +
                                                                         target_lhm[:, 2:] ** 2 + c3)

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
    if reduction == 'none':
        return score
    return {'mean': score.mean,
            'sum': score.sum}[reduction](dim=0)


class MDSILoss(_Loss):
    r"""Creates a criterion that measures Mean Deviation Similarity Index (MDSI) error between the prediction and
    target.

    Args:
        data_range: Value range of input images (usually 1.0 or 255). Default: 1.0
        reduction: Reduction over samples in batch: "mean"|"sum"|"none"
        c1: coefficient to calculate gradient similarity. Default: 140.
        c2: coefficient to calculate gradient similarity. Default: 55.
        c3: coefficient to calculate chromaticity similarity. Default: 550.
        combination: mode to combine gradient similarity and chromaticity similarity: "sum"|"mult".
        alpha: coefficient to combine gradient similarity and chromaticity similarity using summation.
        beta: power to combine gradient similarity with chromaticity similarity using multiplication.
        gamma: to combine gradient similarity and chromaticity similarity using multiplication.
        rho: order of the Minkowski distance
        q: coefficient to adjusts the emphasis of the values in image and MCT
        o: the power pooling applied on the final value of the deviation

    Shape:
        - Input: Required to be 2D (H, W), 3D (C, H, W) or 4D (N, C, H, W). RGB channel order for colour images.
        - Target: Required to be 2D (H, W), 3D (C, H, W) or 4D (N, C, H, W). RGB channel order for colour images.

        Both inputs are supposed to have RGB channels order in accordance with the original approach.
        Nevertheless, the method supports greyscale images, which they are converted to RGB
        by copying the grey channel 3 times.

    Examples::

        >>> loss = MDSILoss(data_range=1.)
        >>> prediction = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> target = torch.rand(3, 3, 256, 256)
        >>> output = loss(prediction, target)
        >>> output.backward()

    References:
        .. [1] Nafchi, Hossein Ziaei and Shahkolaei, Atena and Hedjam, Rachid and Cheriet, Mohamed
           (2016). Mean deviation similarity index: Efficient and reliable full-reference image quality evaluator.
           IEEE Ieee Access,
           4, 5579--5590.
           https://ieeexplore.ieee.org/abstract/document/7556976/,
           :DOI:`10.1109/ACCESS.2016.2604042`
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

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Computation of Mean Deviation Similarity Index (MDSI) as a loss function.
        Both inputs are supposed to have RGB channels order.
        Greyscale images converted to RGB by copying the grey channel 3 times.

        Args:
            prediction: Predicted images. Shape (H, W), (C, H, W) or (N, C, H, W).
            target: Target images. Shape (H, W), (C, H, W) or (N, C, H, W).

        Returns:
            Value of MDSI loss to be minimized. 0 <= MDSI loss <= 1.

        Note:
            Both inputs are supposed to have RGB channels order in accordance with the original approach.
            Nevertheless, the method supports greyscale images, which are converted to RGB by copying the grey
            channel 3 times.
        """
        return 1. - torch.clamp(self.mdsi(prediction=prediction, target=target), min=0., max=1.)
