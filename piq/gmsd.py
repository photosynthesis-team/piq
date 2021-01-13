"""
PyTorch implementation of Gradient Magnitude Similarity Deviation (GMSD)
and Multi-Scale Gradient Magnitude Similarity Deviation (MS-GMSD)
Reference:
    Wufeng Xue et al. Gradient Magnitude Similarity Deviation (2013)
    https://arxiv.org/pdf/1308.3052.pdf
    GRADIENT MAGNITUDE SIMILARITY DEVIATION ON MULTIPLE SCALES (2017)
    http://www.cse.ust.hk/~psander/docs/gradsim.pdf

"""
import torch
from typing import Optional, Union, Tuple, List, cast

import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from piq.utils import _adjust_dimensions, _validate_input
from piq.functional import similarity_map, gradient_map, prewitt_filter, rgb2yiq


def gmsd(prediction: torch.Tensor, target: torch.Tensor, reduction: str = 'mean',
         data_range: Union[int, float] = 1., t: float = 170 / (255. ** 2)) -> torch.Tensor:
    r"""Compute Gradient Magnitude Similarity Deviation
    Inputs supposed to be in range [0, data_range] with RGB channels order for colour images.

    Args:
        prediction: Tensor with shape (H, W), (C, H, W) or (N, C, H, W).
        target: Tensor with shape (H, W), (C, H, W) or (N, C, H, W).
        reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.
        data_range: The difference between the maximum and minimum of the pixel value,
            i.e., if for image x it holds min(x) = 0 and max(x) = 1, then data_range = 1.
            The pixel value interval of both input and output should remain the same.
        t: Constant from the reference paper numerical stability of similarity map.

    Returns:
        gmsd : Gradient Magnitude Similarity Deviation between given tensors.

    References:
        Wufeng Xue et al. Gradient Magnitude Similarity Deviation (2013)
        https://arxiv.org/pdf/1308.3052.pdf
    """

    _validate_input(
        input_tensors=(prediction, target), allow_5d=False, scale_weights=None, data_range=data_range)
    prediction, target = _adjust_dimensions(input_tensors=(prediction, target))

    # Rescale
    prediction = prediction / data_range
    target = target / data_range

    num_channels = prediction.size(1)
    if num_channels == 3:
        prediction = rgb2yiq(prediction)[:, :1]
        target = rgb2yiq(target)[:, :1]
    up_pad = 0
    down_pad = max(prediction.shape[2] % 2, prediction.shape[3] % 2)
    pad_to_use = [up_pad, down_pad, up_pad, down_pad]
    prediction = F.pad(prediction, pad=pad_to_use)
    target = F.pad(target, pad=pad_to_use)

    prediction = F.avg_pool2d(prediction, kernel_size=2, stride=2, padding=0)
    target = F.avg_pool2d(target, kernel_size=2, stride=2, padding=0)

    score = _gmsd(prediction=prediction, target=target, t=t)
    if reduction == 'none':
        return score

    return {'mean': score.mean,
            'sum': score.sum
            }[reduction](dim=0)


def _gmsd(prediction: torch.Tensor, target: torch.Tensor,
          t: float = 170 / (255. ** 2), alpha: float = 0.0) -> torch.Tensor:
    r"""Compute Gradient Magnitude Similarity Deviation
    Both inputs supposed to be in range [0, 1] with RGB channels order.
    Args:
        prediction: Tensor with shape (N, 1, H, W).
        target: Tensor with shape (N, 1, H, W).
        t: Constant from the reference paper numerical stability of similarity map
        alpha: Masking coefficient for similarity masks computation

    Returns:
        gmsd : Gradient Magnitude Similarity Deviation between given tensors.

    References:
        https://arxiv.org/pdf/1308.3052.pdf
    """

    # Compute grad direction
    kernels = torch.stack([prewitt_filter(), prewitt_filter().transpose(-1, -2)])
    pred_grad = gradient_map(prediction, kernels)
    trgt_grad = gradient_map(target, kernels)

    # Compute GMS
    gms = similarity_map(pred_grad, trgt_grad, constant=t, alpha=alpha)
    mean_gms = torch.mean(gms, dim=[1, 2, 3], keepdims=True)
    # Compute GMSD along spatial dimensions. Shape (batch_size )
    score = torch.pow(gms - mean_gms, 2).mean(dim=[1, 2, 3]).sqrt()
    return score


class GMSDLoss(_Loss):
    r"""Creates a criterion that measures Gradient Magnitude Similarity Deviation
    between each element in the input and target.

    Args:
        reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        data_range: The difference between the maximum and minimum of the pixel value,
            i.e., if for image x it holds min(x) = 0 and max(x) = 1, then data_range = 1.
            The pixel value interval of both input and output should remain the same.
        t: Constant from the reference paper numerical stability of similarity map
            
    Reference:
        Wufeng Xue et al. Gradient Magnitude Similarity Deviation (2013)
        https://arxiv.org/pdf/1308.3052.pdf
        
    """

    def __init__(self, reduction: str = 'mean', data_range: Union[int, float] = 1.,
                 t: float = 170 / (255. ** 2)) -> None:
        super().__init__()

        # Generic loss parameters.
        self.reduction = reduction

        # Loss-specific parameters.
        self.data_range = data_range
        self.t = t

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Computation of Gradient Magnitude Similarity Deviation (GMSD) as a loss function.
        Inputs supposed to be in range [0, data_range] with RGB channels order for colour images.

        Args:
            prediction: Tensor with shape (H, W), (C, H, W) or (N, C, H, W).
            target: Tensor with shape (H, W), (C, H, W) or (N, C, H, W).

        Returns:
            Value of GMSD loss to be minimized. 0 <= GMSD loss <= 1.
        """

        return gmsd(prediction=prediction, target=target, reduction=self.reduction, data_range=self.data_range,
                    t=self.t)


def multi_scale_gmsd(prediction: torch.Tensor, target: torch.Tensor, data_range: Union[int, float] = 1.,
                     reduction: str = 'mean',
                     scale_weights: Optional[Union[torch.Tensor, Tuple[float, ...], List[float]]] = None,
                     chromatic: bool = False, alpha: float = 0.5, beta1: float = 0.01, beta2: float = 0.32,
                     beta3: float = 15., t: float = 170) -> torch.Tensor:
    r"""Computation of Multi scale GMSD.
    Inputs supposed to be in range [0, data_range] with RGB channels order for colour images.
    The height and width should be at least 2 ** scales + 1.

    Args:
        prediction: Tensor with shape (H, W), (C, H, W) or (N, C, H, W).
        target: Tensor with shape (H, W), (C, H, W) or (N, C, H, W).
        data_range: The difference between the maximum and minimum of the pixel value,
            i.e., if for image x it holds min(x) = 0 and max(x) = 1, then data_range = 1.
            The pixel value interval of both input and output should remain the same.
        reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        scale_weights: Weights for different scales. Can contain any number of floating point values.
        chromatic: Flag to use MS-GMSDc algorithm from paper.
            It also evaluates chromatic components of the image. Default: True
        alpha: Masking coefficient. See [1] for details.
        beta1: Algorithm parameter. Weight of chromatic component in the loss.
        beta2: Algorithm parameter. Small constant, see [1].
        beta3: Algorithm parameter. Small constant, see [1].
        t: Constant from the reference paper numerical stability of similarity map

    Returns:
        Value of MS-GMSD. 0 <= GMSD loss <= 1.
    """
    _validate_input(
        input_tensors=(prediction, target), allow_5d=False, scale_weights=scale_weights, data_range=data_range)
    prediction, target = _adjust_dimensions(input_tensors=(prediction, target))
    
    # Rescale
    prediction = prediction / data_range * 255
    target = target / data_range * 255
    
    # Values from the paper
    if scale_weights is None:
        scale_weights = torch.tensor([0.096, 0.596, 0.289, 0.019])
    else:
        # Normalize scale weights
        scale_weights = torch.tensor(scale_weights) / torch.tensor(scale_weights).sum()
    scale_weights = cast(torch.Tensor, scale_weights).to(prediction)

    # Check that input is big enough
    num_scales = scale_weights.size(0)
    min_size = 2 ** num_scales + 1

    if prediction.size(-1) < min_size or prediction.size(-2) < min_size:
        raise ValueError(f'Invalid size of the input images, expected at least {min_size}x{min_size}.')

    num_channels = prediction.size(1)
    if num_channels == 3:
        prediction = rgb2yiq(prediction)
        target = rgb2yiq(target)

    ms_gmds = []
    for scale in range(num_scales):
        if scale > 0:
            # Average by 2x2 filter and downsample
            up_pad = 0
            down_pad = max(prediction.shape[2] % 2, prediction.shape[3] % 2)
            pad_to_use = [up_pad, down_pad, up_pad, down_pad]
            prediction = F.pad(prediction, pad=pad_to_use)
            target = F.pad(target, pad=pad_to_use)
            prediction = F.avg_pool2d(prediction, kernel_size=2, padding=0)
            target = F.avg_pool2d(target, kernel_size=2, padding=0)

        score = _gmsd(prediction[:, :1], target[:, :1], t=t, alpha=alpha)
        ms_gmds.append(score)

    # Stack results in different scales and multiply by weight
    ms_gmds_val = scale_weights.view(1, num_scales) * (torch.stack(ms_gmds, dim=1) ** 2)

    # Sum and take sqrt per-image
    ms_gmds_val = torch.sqrt(torch.sum(ms_gmds_val, dim=1))

    # Shape: (batch_size, )
    score = ms_gmds_val

    if chromatic:
        assert prediction.size(1) == 3, "Chromatic component can be computed only for RGB images!"

        prediction_iq = prediction[:, 1:]
        target_iq = target[:, 1:]

        rmse_iq = torch.sqrt(torch.mean((prediction_iq - target_iq) ** 2, dim=[2, 3]))
        rmse_chrome = torch.sqrt(torch.sum(rmse_iq ** 2, dim=1))
        gamma = 2 / (1 + beta2 * torch.exp(-beta3 * ms_gmds_val)) - 1

        score = gamma * ms_gmds_val + (1 - gamma) * beta1 * rmse_chrome

    if reduction == 'none':
        return score

    return {'mean': score.mean,
            'sum': score.sum
            }[reduction](dim=0)


class MultiScaleGMSDLoss(_Loss):
    r"""Creates a criterion that measures multi scale Gradient Magnitude Similarity Deviation
    between each element in the input :math:`x` and target :math:`y`.

    Args:
        reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        data_range: The difference between the maximum and minimum of the pixel value,
            i.e., if for image x it holds min(x) = 0 and max(x) = 1, then data_range = 1.
            The pixel value interval of both input and output should remain the same.
        scale_weights: Weights for different scales. Can contain any number of floating point values.
            By defualt weights are initialized with values from the paper.
        chromatic: Flag to use MS-GMSDc algorithm from paper.
            It also evaluates chromatic components of the image. Default: True
        beta1: Algorithm parameter. Weight of chromatic component in the loss.
        beta2: Algorithm parameter. Small constant, see [1].
        beta3: Algorithm parameter. Small constant, see [1].
        t: Constant from the reference paper numerical stability of similarity map

    Reference:
        [1] GRADIENT MAGNITUDE SIMILARITY DEVIATION ON MULTIPLE SCALES (2017)
            http://www.cse.ust.hk/~psander/docs/gradsim.pdf
    """

    def __init__(self, reduction: str = 'mean', data_range: Union[int, float] = 1.,
                 scale_weights: Optional[Union[torch.Tensor, Tuple[float, ...], List[float]]] = None,
                 chromatic: bool = False, alpha: float = 0.5, beta1: float = 0.01, beta2: float = 0.32,
                 beta3: float = 15., t: float = 170) -> None:
        super().__init__()

        # Generic loss parameters.
        self.reduction = reduction

        # Loss-specific parameters.
        self.data_range = data_range

        self.scale_weights = scale_weights
        self.chromatic = chromatic
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.t = t
            
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Computation of Multi Scale GMSD index as a loss function.
        Inputs supposed to be in range [0, data_range] with RGB channels order for colour images.
        The height and width should be at least 2 ** scales + 1.

        Args:
            prediction: Tensor with shape (H, W), (C, H, W) or (N, C, H, W).
            target: Tensor with shape (H, W), (C, H, W) or (N, C, H, W).

        Returns:
            Value of MS-GMSD loss to be minimized. 0 <= MS-GMSD loss <= 1.
        """
        return multi_scale_gmsd(prediction=prediction, target=target, data_range=self.data_range,
                                reduction=self.reduction, chromatic=self.chromatic, alpha=self.alpha, beta1=self.beta1,
                                beta2=self.beta2, beta3=self.beta3, scale_weights=self.scale_weights, t=self.t)
