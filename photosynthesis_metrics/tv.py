r"""Implemetation of Total Variation metric, based on article
 remi.flamary.com/demos/proxtv.html and www.wikiwand.com/en/Total_variation_denoising
"""

import functools

import torch
from torch.nn.modules.loss import _Loss

from photosynthesis_metrics.utils import _adjust_dimensions, _validate_input


def _compute_tv(x: torch.Tensor, size_average: bool = True, reduction_type: str = 'l2') -> torch.Tensor:
    r"""Compute Total Variation metric
    See www.wikiwand.com/en/Total_variation_denoising for additional info about reduction types

    Args:
        x: Tensor of shape :math:`(N, C, H, W)` holding an input image.
        size_average: If size_average=True, total variation of all images will be averaged as a scalar.
        reduction_type: {'l1', 'l2', 'l2_squared'}, defines which type of norm to implement, isotropic  or anisotropic.

    Returns:
        tv : Total variation of a given tensor
    """
    if reduction_type == 'l1':
        w_variance = torch.sum(torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]), dim=[1, 2, 3])
        h_variance = torch.sum(torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]), dim=[1, 2, 3])
        tv_val = (h_variance + w_variance)
    elif reduction_type == 'l2':
        w_variance = torch.sum(torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2), dim=[1, 2, 3])
        h_variance = torch.sum(torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2), dim=[1, 2, 3]) 
        tv_val = torch.sqrt(h_variance + w_variance)
    elif reduction_type == 'l2_squared':
        w_variance = torch.sum(torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2), dim=[1, 2, 3])
        h_variance = torch.sum(torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2), dim=[1, 2, 3])  
        tv_val = (h_variance + w_variance)
    else:
        raise ValueError("Incorrect reduction type, should be one of {'l1', 'l2', 'l2_squared'}")

    if size_average:
        return tv_val.mean()
    else:
        return tv_val


def total_variation(x: torch.Tensor, size_average: bool = True, reduction_type: str = 'l2') -> torch.Tensor:
    r"""Interface of Total Variation.

    Args:
        x: Batch of images. Required to be 4D, channels first (N,C,H,W).
        size_average: If size_average=True, total_variation of all images will be averaged as a scalar.
        reduction_type: one of {'l1', 'l2', 'l2_squared'}

    Returns:
        Total variation of a given tensor

    References:
        remi.flamary.com/demos/proxtv.html and www.wikiwand.com/en/Total_variation_denoising
    """
    _validate_input(x=x, y=x)
    
    tv = _compute_tv(x=x, 
                     size_average=size_average, 
                     reduction_type=reduction_type)
    return tv


class TVLoss(_Loss):
    r"""Creates a criterion that measures the total variation error between
    each element in the input :math:`x` and target :math:`y`.
    See https://remi.flamary.com/demos/proxtv.html for reference.

    If :attr:`reduction_type` set to ``'l2'`` loss can be described as:

    .. math::
        TV(x) = \sum_{N}\sqrt{\sum_{H, W, C}(|x_{:, :, i+1, j} - x_{:, :, i, j}|^2 +
        |x_{:, :, i, j+1} - x_{:, :, i, j}|^2)}
                                            
    Else if :attr:`reduction_type` set to ``'l1'``:

    .. math::
        TV(x) = \sum_{N}\sum_{H, W, C}(|x_{:, :, i+1, j} - x_{:, :, i, j}| + 
        |x_{:, :, i, j+1} - x_{:, :, i, j}|) $$

    where :math:`N` is the batch size, `C` is the channel size.

    .. math::
        TVLoss(x, y) = |TV(x) - TV(y)|

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    Args:
        size_average: If size_average=True, total_variation of all images will be averaged as a scalar.
        reduction_type: one of {'l1', 'l2', 'l2_squared'}
        reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
    Examples::

        >>> loss = TVLoss()
        >>> prediction = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> target = torch.rand(3, 3, 256, 256)
        >>> output = loss(prediction, target, max_val=1.)
        >>> output.backward()
    """

    def __init__(self, size_average: bool = True, reduction_type: str = 'l2', reduction: str = 'mean'):
        super(TVLoss, self).__init__()

        # Generic loss parameters.
        self.reduction = reduction

        # Loss-specific parameters.
        self.tv_func = functools.partial(_compute_tv, size_average=size_average, reduction_type=reduction_type)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Computation of Total Variation (TV) index as a loss function.

        Args:
            prediction: Tensor of prediction of the network.
            target: Reference tensor.
            
        Returns:
            Value of TV loss to be minimized. 
        """
        prediction, target = _adjust_dimensions(x=prediction, y=target)
        _validate_input(x=prediction, y=target)

        prediction_tv = self.tv_func(prediction)
        target_tv = self.tv_func(target)
        
        res = torch.abs(prediction_tv - target_tv)

        if self.reduction != 'none':
            res = torch.mean(res) if self.reduction == 'mean' else torch.sum(res)

        return res
