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
from typing import Optional, Union, Tuple, List

import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from piq.utils import _adjust_dimensions, _validate_input
from piq.functional import similarity_map, gradient_map, prewitt_filter


def _gmsd(prediction: torch.Tensor, target: torch.Tensor,
          reduction: Optional[str] = 'mean') -> torch.Tensor:
    r"""Compute Gradient Magnitude Similarity Deviation
    Both inputs supposed to be in range [0, 1] with RGB order.
    Args:
        prediction: Tensor of shape :math:`(N, C, H, W)` holding an distorted image.
        target: Tensor of shape :math:`(N, C, H, W)` holding an target image
        reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.

    Returns:
        gmsd : Gradient Magnitude Similarity Deviation between given tensors.

    References:
        https://arxiv.org/pdf/1308.3052.pdf
    """
    # Constant for numerical stability
    EPS: float = 0.0026

    # Convert RGB image to YCbCr and take luminance: Y = 0.299 R + 0.587 G + 0.114 B
    num_channels = prediction.size(1)
    if num_channels == 3:
        prediction = 0.299 * prediction[:, 0, :, :] + 0.587 * prediction[:, 1, :, :] + 0.114 * prediction[:, 2, :, :]
        target = 0.299 * target[:, 0, :, :] + 0.587 * target[:, 1, :, :] + 0.114 * target[:, 2, :, :]

        # Add channel dimension
        prediction = prediction[:, None, :, :]
        target = target[:, None, :, :]
    
    # Compute grad direction
    kernels = torch.stack([prewitt_filter(), prewitt_filter().transpose(-1, -2)])
    pred_grad = gradient_map(prediction, kernels)
    trgt_grad = gradient_map(target, kernels)

    # Compute GMS
    gms = similarity_map(pred_grad, trgt_grad, EPS)
    mean_gms = torch.mean(gms, dim=[1, 2, 3], keepdims=True)

    # Compute GMSD along spatial dimensions. Shape (batch_size )
    score = torch.pow(gms - mean_gms, 2).mean(dim=[1, 2, 3]).sqrt()
    
    if reduction == 'none':
        return score

    return {'mean': score.mean,
            'sum': score.sum
            }[reduction](dim=0)


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
            
    Reference:
        Wufeng Xue et al. Gradient Magnitude Similarity Deviation (2013)
        https://arxiv.org/pdf/1308.3052.pdf
        
    """

    def __init__(self, reduction: str = 'mean', data_range: Union[int, float] = 1.) -> None:
        super().__init__()

        # Generic loss parameters.
        self.reduction = reduction

        # Loss-specific parameters.
        self.data_range = data_range

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Computation of Gradient Magnitude Similarity Deviation (GMSD) as a loss function.

        Args:
            prediction: Tensor of prediction of the network.
            target: Reference tensor.

        Returns:
            Value of GMSD loss to be minimized. 0 <= GMSD loss <= 1.
        """
        _validate_input(input_tensors=(prediction, target), allow_5d=False, scale_weights=None)
        prediction, target = _adjust_dimensions(input_tensors=(prediction, target))

        return self.compute_metric(prediction, target)
    
    def compute_metric(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        
        if self.data_range == 255:
            prediction = prediction / 255.
            target = target / 255.

        # Average by 2x2 filter and downsample
        padding = (prediction.shape[2] % 2, prediction.shape[3] % 2)
        prediction = F.avg_pool2d(prediction, kernel_size=2, stride=2, padding=padding)
        target = F.avg_pool2d(target, kernel_size=2, stride=2, padding=padding)
        
        score = _gmsd(
            prediction, target, reduction=self.reduction)

        return score
    

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
        chromatic: Flag to use MS-GMSDc algorithm from paper.
            It also evaluates chromatic components of the image. Default: True
        beta1: Algorithm parameter. Weight of chromatic component in the loss.
        beta2: Algorithm parameter. Small constant, see [1].
        beta3: Algorithm parameter. Small constant, see [1].
        
    Reference:
        [1] GRADIENT MAGNITUDE SIMILARITY DEVIATION ON MULTIPLE SCALES (2017)
            http://www.cse.ust.hk/~psander/docs/gradsim.pdf
    """

    def __init__(self, reduction: str = 'mean', data_range: Union[int, float] = 1.,
                 scale_weights: Optional[Union[Tuple[float], List[float]]] = None,
                 chromatic: bool = False, beta1: float = 0.01, beta2: float = 0.32,
                 beta3: float = 15.) -> None:
        super().__init__()

        # Generic loss parameters.
        self.reduction = reduction

        # Loss-specific parameters.
        self.data_range = data_range
        
        # Values from the paper
        if scale_weights is None:
            self.scale_weights = torch.tensor([0.096, 0.596, 0.289, 0.019])
        else:
            # Normalize scale weights
            self.scale_weights = torch.tensor(scale_weights) / torch.tensor(scale_weights).sum()

        self.chromatic = chromatic
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
            
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Computation of Multi scale GMSD as a loss function.

        Args:
            prediction: Tensor of prediction of the network.
            target: Reference tensor.

        Returns:
            Value of GMSD loss to be minimized. 0 <= GMSD loss <= 1.
        """
        _validate_input(input_tensors=(prediction, target), allow_5d=False, scale_weights=self.scale_weights)
        prediction, target = _adjust_dimensions(input_tensors=(prediction, target))

        return self.compute_metric(prediction, target)
    
    def compute_metric(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Check that input is big enough
        num_scales = self.scale_weights.size(0)
        min_size = 2 ** num_scales + 1

        if prediction.size(-1) < min_size or prediction.size(-2) < min_size:
            raise ValueError(f'Invalid size of the input images, expected at least {min_size}x{min_size}.')
        
        if self.data_range == 255:
            prediction = prediction / 255.
            target = target / 255.
        
        scale_weights = self.scale_weights.to(prediction)
        ms_gmds = []
        for scale in range(num_scales):
            if scale > 0:
                # Average by 2x2 filter and downsample
                padding = (prediction.shape[2] % 2, prediction.shape[3] % 2)
                prediction = F.avg_pool2d(prediction, kernel_size=2, padding=padding)
                target = F.avg_pool2d(target, kernel_size=2, padding=padding)
        
            score = _gmsd(prediction, target, reduction='none')
            ms_gmds.append(score)
        
        # Stack results in different scales and multiply by weight
        ms_gmds_val = scale_weights.view(1, num_scales) * (torch.stack(ms_gmds, dim=1) ** 2)
  
        # Sum and take sqrt per-image
        ms_gmds_val = torch.sqrt(torch.sum(ms_gmds_val, dim=1))
        
        # Shape: (batch_size, )
        score = ms_gmds_val
        
        if self.chromatic:
            assert prediction.size(1) == 3, "Chromatic component can be computed only for RGB images!"
            
            # Convert to YIQ color space https://en.wikipedia.org/wiki/YIQ
            iq_weights = torch.tensor([[0.5959, -0.2746, -0.3213], [0.2115, -0.5227, 0.3112]]).t().to(prediction)
            prediction_iq = torch.matmul(prediction.permute(0, 2, 3, 1), iq_weights).permute(0, 3, 1, 2)
            target_iq = torch.matmul(target.permute(0, 2, 3, 1), iq_weights).permute(0, 3, 1, 2)
            
            rmse_iq = torch.sqrt(torch.mean((prediction_iq - target_iq) ** 2, dim=[2, 3]))
            rmse_chrome = torch.sqrt(torch.sum(rmse_iq ** 2, dim=1))
            gamma = 2 / (1 + self.beta2 * torch.exp(-self.beta3 * ms_gmds_val)) - 1
            
            score = gamma * ms_gmds_val + (1 - gamma) * self.beta1 * rmse_chrome
            
        if self.reduction == 'none':
            return score

        return {'mean': score.mean,
                'sum': score.sum
                }[self.reduction](dim=0)
