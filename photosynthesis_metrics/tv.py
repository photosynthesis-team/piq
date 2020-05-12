r"""Implemetation of Total Variation metric, based on article
 remi.flamary.com/demos/proxtv.html and www.wikiwand.com/en/Total_variation_denoising
"""

import torch
from torch.nn.modules.loss import _Loss

from photosynthesis_metrics.utils import _adjust_dimensions, _validate_input


def total_variation(
    x: torch.Tensor, size_average: bool = True, reduction_type: str = 'l2'
    ) -> torch.Tensor:
    r"""Compute Total Variation metric

    Args:
        x: Tensor of shape :math:`(N, C, H, W)` holding an input image.
        size_average: If size_average=True, total variation of all images will be averaged as a scalar.
        reduction_type: {'l1', 'l2', 'l2_squared'}, defines which type of norm to implement, isotropic  or anisotropic.

    Returns:
        tv : Total variation of a given tensor
    
    References:
        https://www.wikiwand.com/en/Total_variation_denoising
        https://remi.flamary.com/demos/proxtv.html
    """
    if reduction_type == 'l1':
        w_variance = torch.sum(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]), dim=[1, 2, 3])
        h_variance = torch.sum(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]), dim=[1, 2, 3])
        tv_val = (h_variance + w_variance)
    elif reduction_type == 'l2':
        w_variance = torch.sum(torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2), dim=[1, 2, 3])
        h_variance = torch.sum(torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2), dim=[1, 2, 3])
        tv_val = torch.sqrt(h_variance + w_variance)
    elif reduction_type == 'l2_squared':
        w_variance = torch.sum(torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2), dim=[1, 2, 3])
        h_variance = torch.sum(torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2), dim=[1, 2, 3])
        tv_val = (h_variance + w_variance)
    else:
        raise ValueError("Incorrect reduction type, should be one of {'l1', 'l2', 'l2_squared'}")

    if size_average:
        return tv_val.mean()
    else:
        return tv_val


class TVLoss(_Loss):
    r"""Creates a criterion that measures the total variation error between
    each element in the input :math:`x` and target :math:`y`.


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

    References:
        https://www.wikiwand.com/en/Total_variation_denoising
        https://remi.flamary.com/demos/proxtv.html
    """

    def __init__(self, size_average: bool = True, reduction_type: str = 'l2', reduction: str = 'mean'):
        super().__init__()

        self.size_average = size_average
        self.reduction_type = reduction_type
        self.reduction = reduction

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

        return self.compute_metric(prediction, target)

    def compute_metric(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction_tv = total_variation(
            prediction,
            size_average=self.size_average,
            reduction_type=self.reduction_type
        )
        target_tv = total_variation(
            target,
            size_average=self.size_average,
            reduction_type=self.reduction_type
        )
        score = torch.abs(prediction_tv - target_tv)

        if self.reduction == 'mean':
            score = torch.mean(score)
        elif self.reduction == 'sum':
            score = torch.sum(score)
        return score
