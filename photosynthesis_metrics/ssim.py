r""" This module implements Structural Similarity (SSIM) index in PyTorch.

Implementation of classes and functions from this module are inspired by Gongfan Fang's (@VainF) implementation:
https://github.com/VainF/pytorch-msssim

and implementation of one of pull requests to the PyTorch by Kangfu Mei (@MKFMIKU):
https://github.com/pytorch/pytorch/pull/22289/files
"""
from typing import List, Optional, Tuple, Union

import torch
import torch.nn._reduction as _Reduction
import torch.nn.functional as f
from torch.nn.modules.loss import _Loss

from photosynthesis_metrics.utils import _adjust_dimensions, _validate_input

from .utils import _adjust_dimensions, _validate_input


def ssim(x: torch.Tensor, y: torch.Tensor, kernel_size: int = 11, kernel_sigma: float = 1.5,
         data_range: Union[int, float] = 255, size_average: bool = True, full: bool = False,
         k1: float = 0.01, k2: float = 0.03) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Interface of Structural Similarity (SSIM) index.

    Args:
        x: Batch of images. Required to be 4D, channels first (N,C,H,W).
        y: Batch of images. Required to be 4D, channels first (N,C,H,W).
        kernel_size: The side-length of the sliding window used in comparison. Must be an odd value.
        kernel_sigma: Sigma of normal distribution.
        data_range: Value range of input images (usually 1.0 or 255).
        size_average: If size_average=True, ssim of all images will be averaged as a scalar.
        full: Return sc or not.
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Value of Structural Similarity (SSIM) index.

    References:
        .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
           (2004). Image quality assessment: From error visibility to
           structural similarity. IEEE Transactions on Image Processing,
           13, 600-612.
           https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
           :DOI:`10.1109/TIP.2003.819861`
    """
    _validate_input(x=x, y=y, kernel_size=kernel_size, scale_weights=None)

    kernel = _fspecial_gauss_1d(kernel_size, kernel_sigma)
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)

    ssim_val, cs = _compute_ssim(x=x,
                                 y=y,
                                 kernel=kernel,
                                 data_range=data_range,
                                 size_average=False,
                                 full=True,
                                 k1=k1,
                                 k2=k2)
    if size_average:
        ssim_val = ssim_val.mean()
        cs = cs.mean()

    if full:
        return ssim_val, cs

    return ssim_val


class SSIMLoss(_Loss):
    r"""Creates a criterion that measures the structural similarity index error between
    each element in the input :math:`x` and target :math:`y`.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        SSIM = \{ssim_1,\dots,ssim_{N \times C}\}, \quad
        ssim_{l}(x, y) = \frac{(2 \mu_x \mu_y + c_1) (2 \sigma_{xy} + c_2)}
        {(\mu_x^2 +\mu_y^2 + c_1)(\sigma_x^2 +\sigma_y^2 + c_2)},

    where :math:`N` is the batch size, `C` is the channel size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        SSIMLoss(x, y) =
        \begin{cases}
            \operatorname{mean}(SSIM), &  \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(SSIM),  &  \text{if reduction} = \text{'sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    The sum operation still operates over all the elements, and divides by :math:`n`.

    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

    Args:
        kernel_size: By default, the mean and covariance of a pixel is obtained
            by convolution with given filter_size.
        kernel_sigma: Standard deviation for Gaussian kernel.
        k1: Coefficient related to c1 in the above equation.
        k2: Coefficient related to c2 in the above equation.
        size_average: Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce: Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        data_range: The difference between the maximum and minimum of the pixel value,
            i.e., if for image x it holds min(x) = 0 and max(x) = 1, then max_val = 1.
            The pixel value interval of both input and output should remain the same.

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means, any number of additional dimensions
        - Target: :math:`(N, *)`, same shape as the input

    Examples::

        >>> loss = SSIMLoss()
        >>> prediction = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> target = torch.rand(3, 3, 256, 256)
        >>> output = loss(prediction, target, max_val=1.)
        >>> output.backward()
    """
    __constants__ = ['filter_size', 'k1', 'k2', 'sigma', 'kernel', 'reduction']

    def __init__(self, kernel_size: int = 11, kernel_sigma: float = 1.5, k1: float = 0.01, k2: float = 0.03,
                 size_average: Optional[bool] = None, reduce: Optional[bool] = None, reduction: str = 'mean',
                 data_range: Union[int, float] = 1.) -> None:
        super(SSIMLoss, self).__init__(size_average, reduce, reduction)

        # Generic loss parameters.
        self.size_average = size_average
        self.reduce = reduce
        if size_average is not None or reduce is not None:
            reduction = _Reduction.legacy_get_string(size_average, reduce)

        self.reduction = reduction

        # Loss-specific parameters.
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self.k1 = k1
        self.k2 = k2
        self.data_range = data_range

        # Cash kernel between calls.
        self.kernel = _fspecial_gauss_1d(kernel_size, kernel_sigma)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Computation of Structural Similarity (SSIM) index as a loss function.

        Args:
            prediction: Tensor of prediction of the network.
            target: Reference tensor.

        Returns:
            Value of SSIM loss to be minimized. 0 <= SSIM loss <= 1.
        """
        prediction, target = _adjust_dimensions(x=prediction, y=target)
        kernel = self.kernel.repeat(prediction.shape[1], 1, 1, 1)
        kernel = kernel.to(device=prediction.device)

        ret = _compute_ssim(x=prediction,
                            y=target,
                            kernel=kernel,
                            data_range=self.data_range,
                            size_average=False,
                            full=False,
                            k1=self.k1,
                            k2=self.k2)

        if self.reduction != 'none':
            ret = torch.mean(ret) if self.reduction == 'mean' else torch.sum(ret)

        return ret


def multi_scale_ssim(x: torch.Tensor, y: torch.Tensor, kernel_size: int = 11, kernel_sigma: float = 1.5,
                     data_range: Union[int, float] = 255, size_average: bool = True,
                     scale_weights: Optional[Union[Tuple[float], List[float]]] = None, k1=0.01, k2=0.03) -> torch.Tensor:
    r""" Interface of Multi-scale Structural Similarity (MS-SSIM) index.

    Args:
        x: Batch of images. Required to be 4D, channels first (N,C,H,W).
        y: Batch of images. Required to be 4D, channels first (N,C,H,W).
        kernel_size: The side-length of the sliding window used in comparison. Must be an odd value.
        kernel_sigma: Sigma of normal distribution.
        data_range: Value range of input images (usually 1.0 or 255).
        size_average: If size_average=True, ssim of all images will be averaged as a scalar.
        scale_weights: Weights for different scales. Must contain 4 floating point values.
            If None, default weights from the paper [1] will be used.
            Default weights: (0.0448, 0.2856, 0.3001, 0.2363, 0.1333).
        k1: Algorithm parameter, K1 (small constant, see [2]).
        k2: Algorithm parameter, K2 (small constant, see [2]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Value of Multi-scale Structural Similarity (MS-SSIM) index.

    References:
        .. [1] Wang, Z., Simoncelli, E. P., Bovik, A. C. (2003).
           Multi-scale Structural Similarity for Image Quality Assessment.
           IEEE Asilomar Conference on Signals, Systems and Computers, 37,
           https://ieeexplore.ieee.org/document/1292216
           :DOI:`10.1109/ACSSC.2003.1292216`
        .. [2] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
           (2004). Image quality assessment: From error visibility to
           structural similarity. IEEE Transactions on Image Processing,
           13, 600-612.
           https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
           :DOI:`10.1109/TIP.2003.819861`
    """
    _validate_input(x=x, y=y, kernel_size=kernel_size, scale_weights=scale_weights)

    if scale_weights is None:
        scale_weights_from_ms_ssim_paper = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        scale_weights = scale_weights_from_ms_ssim_paper

    scale_weights_tensor = torch.tensor(scale_weights).to(x.device, dtype=x.dtype)
    kernel = _fspecial_gauss_1d(kernel_size, kernel_sigma)
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)

    msssim_val = _compute_multi_scale_ssim(x=x,
                                           y=y,
                                           data_range=data_range,
                                           kernel=kernel,
                                           scale_weights_tensor=scale_weights_tensor,
                                           k1=k1,
                                           k2=k2)

    if size_average:
        msssim_val = msssim_val.mean()

    return msssim_val


class MultiScaleSSIMLoss(_Loss):
    r"""Creates a criterion that measures the multi-scale structural similarity index error between
    each element in the input :math:`x` and target :math:`y`.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        MSSIM = \{mssim_1,\dots,mssim_{N \times C}\}, \quad
        mssim_{l}(x, y) = \frac{(2 \mu_{x,m} \mu_{y,m} + c_1) }
        {(\mu_{x,m}^2 +\mu_{y,m}^2 + c_1)} \prod_{j=1}^{m - 1}
        \frac{(2 \sigma_{xy,j} + c_2)}{(\sigma_{x,j}^2 +\sigma_{y,j}^2 + c_2)}

    where :math:`N` is the batch size, `C` is the channel size, `m` is the scale level (Default: 5).
    If :attr:`reduction` is not ``'none'``(default ``'mean'``), then:

    .. math::
        MultiscaleSSIMLoss(x, y) =
        \begin{cases}
            \operatorname{mean}(MSSIM), &  \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(MSSIM),  &  \text{if reduction} = \text{'sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    The sum operation still operates over all the elements, and divides by :math:`n`.

    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

   Args:
        kernel_size: By default, the mean and covariance of a pixel is obtained
            by convolution with given filter_size.
        kernel_sigma: Standard deviation for Gaussian kernel.
        k1: Coefficient related to c1 in the above equation.
        k2: Coefficient related to c2 in the above equation.
        scale_weights:  Weights for different scales. Must contain 4 floating point values.
            If None, default weights from the paper [1] will be used.
            Default weights: (0.0448, 0.2856, 0.3001, 0.2363, 0.1333).
        size_average: Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce: Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        data_range: The difference between the maximum and minimum of the pixel value,
            i.e., if for image x it holds min(x) = 0 and max(x) = 1, then max_val = 1.
            The pixel value interval of both input and output should remain the same.


    Shape:
        - Input: :math:`(N, *)` where :math:`*` means, any number of additional dimensions
        - Target: :math:`(N, *)`, same shape as the input

    Examples::

        >>> loss = MultiScaleSSIMLoss()
        >>> input = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> target = torch.rand(3, 3, 256, 256)
        >>> output = loss(input, target, max_val=1.)
        >>> output.backward()
    """
    __constants__ = ['filter_size', 'k1', 'k2', 'sigma', 'kernel', 'reduction']

    def __init__(self, kernel_size: int = 11, kernel_sigma: float = 1.5, k1: float = 0.01, k2: float = 0.03,
                 scale_weights: Optional[Union[Tuple[float], List[float]]] = None, size_average: Optional[bool] = None,
                 reduce: Optional[bool] = None, reduction: str = 'mean', data_range: Union[int, float] = 1.) -> None:
        super(MultiScaleSSIMLoss, self).__init__(size_average, reduce, reduction)

        # Generic loss parameters.
        self.size_average = size_average
        self.reduce = reduce
        if size_average is not None or reduce is not None:
            reduction = _Reduction.legacy_get_string(size_average, reduce)

        self.reduction = reduction

        # Loss-specific parameters.
        if scale_weights is None:
            scale_weights_from_ms_ssim_paper = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
            scale_weights = scale_weights_from_ms_ssim_paper

        self.scale_weights_tensor = torch.tensor(scale_weights)
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self.k1 = k1
        self.k2 = k2
        self.data_range = data_range

        # Cash kernel between calls.
        self.kernel = _fspecial_gauss_1d(kernel_size, kernel_sigma)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Computation of Multi-scale Structural Similarity (MS-SSIM) index as a loss function.


        Args:
            prediction: Tensor of prediction of the network.
            target: Reference tensor.

        Returns:
            Value of MS-SSIM loss to be minimized. 0 <= MS-SSIM loss <= 1.
        """
        prediction, target = _adjust_dimensions(x=prediction, y=target)
        kernel = self.kernel.repeat(prediction.shape[1], 1, 1, 1)
        scale_weights_tensor = self.scale_weights_tensor.to(device=prediction.device)

        ret = _compute_multi_scale_ssim(x=prediction,
                                        y=target,
                                        data_range=self.data_range,
                                        kernel=kernel,
                                        scale_weights_tensor=scale_weights_tensor,
                                        k1=self.k1,
                                        k2=self.k2)

        if self.reduction != 'none':
            ret = torch.mean(ret) if self.reduction == 'mean' else torch.sum(ret)

        return ret


def _fspecial_gauss_1d(size: int, sigma: float) -> torch.Tensor:
    r""" Creates a 1-D gauss kernel.

    Args:
        size: The size of gauss kernel.
        sigma: Sigma of normal distribution.

    Returns:
        1D Gauss kernel.
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size//2

    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def _compute_ssim(x: torch.Tensor, y: torch.Tensor, kernel: torch.Tensor, data_range: Union[float, int] = 255,
                  size_average: bool = True, full: bool = False, k1: float = 0.01, k2: float = 0.03) \
        -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Calculate Structural Similarity (SSIM) index for X and Y.

    Args:
        x: Batch of images, (N,C,H,W).
        y: Batch of images, (N,C,H,W).
        kernel: 1-D gauss kernel.
        data_range: Value range of input images (usually 1.0 or 255).
        size_average: If size_average=True, ssim of all images will be averaged as a scalar.
        full: Return sc or not.
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Value of Structural Similarity (SSIM) index.
    """
    c1 = (k1 * data_range)**2
    c2 = (k2 * data_range)**2

    kernel = kernel.to(x.device, dtype=x.dtype)

    mu1 = _gaussian_filter(x, kernel)
    mu2 = _gaussian_filter(y, kernel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    compensation = 1.0
    sigma1_sq = compensation * (_gaussian_filter(x * x, kernel) - mu1_sq)
    sigma2_sq = compensation * (_gaussian_filter(y * y, kernel) - mu2_sq)
    sigma12 = compensation * (_gaussian_filter(x * y, kernel) - mu1_mu2)

    # Set alpha = beta = gamma = 1.
    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map

    if size_average:
        ssim_val = ssim_map.mean()
        cs = cs_map.mean()
    else:
        # Reduce along CHW.
        ssim_val = ssim_map.mean(-1).mean(-1).mean(-1)
        cs = cs_map.mean(-1).mean(-1).mean(-1)

    if full:
        return ssim_val, cs

    return ssim_val


def _compute_multi_scale_ssim(x: torch.Tensor, y: torch.Tensor, data_range: Union[int, float], kernel: torch.Tensor,
                              scale_weights_tensor: torch.Tensor, k1: float, k2: float) -> torch.Tensor:
    levels = scale_weights_tensor.shape[0]
    mcs = []
    ssim_val = None
    for _ in range(levels):
        ssim_val, cs = _compute_ssim(x, y,
                                     kernel=kernel,
                                     data_range=data_range,
                                     size_average=False,
                                     full=True,
                                     k1=k1,
                                     k2=k2)
        mcs.append(cs)

        padding = (x.shape[2] % 2, x.shape[3] % 2)
        x = f.avg_pool2d(x, kernel_size=2, padding=padding)
        y = f.avg_pool2d(y, kernel_size=2, padding=padding)

    # mcs, (level, batch)
    mcs = torch.stack(mcs, dim=0)

    # weights, (level)
    msssim_val = torch.prod((mcs[:-1] ** scale_weights_tensor[:-1].unsqueeze(1)) *
                            (ssim_val ** scale_weights_tensor[-1]), dim=0)

    return msssim_val


def _gaussian_filter(to_blur: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    r""" Blur input with 1-D kernel.

    Args:
        to_blur: A batch of tensors to be blured.
        window: 1-D gauss kernel.

    Returns:
        A batch of blurred tensors.
    """
    _, n_channels, _, _ = to_blur.shape
    out = f.conv2d(to_blur, window, stride=1, padding=0, groups=n_channels)
    out = f.conv2d(out, window.transpose(2, 3), stride=1, padding=0, groups=n_channels)
    return out
