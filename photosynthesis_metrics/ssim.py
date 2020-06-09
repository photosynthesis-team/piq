r""" This module implements Structural Similarity (SSIM) index in PyTorch.

Implementation of classes and functions from this module are inspired by Gongfan Fang's (@VainF) implementation:
https://github.com/VainF/pytorch-msssim

and implementation of one of pull requests to the PyTorch by Kangfu Mei (@MKFMIKU):
https://github.com/pytorch/pytorch/pull/22289/files
"""
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as f
from torch.nn.modules.loss import _Loss

from photosynthesis_metrics.utils import _adjust_dimensions, _validate_input


def ssim(x: torch.Tensor, y: torch.Tensor, kernel_size: int = 11, kernel_sigma: float = 1.5,
         data_range: Union[int, float] = 255, size_average: bool = True, full: bool = False,
         k1: float = 0.01, k2: float = 0.03) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Interface of Structural Similarity (SSIM) index.
    Args:
        x: Batch of images. Required to be 2D (H, W), 3D (C,H,W), 4D (N,C,H,W) or 5D (N,C,H,W,2), channels first.
        y: Batch of images. Required to be 2D (H, W), 3D (C,H,W) 4D (N,C,H,W) or 5D (N,C,H,W,2), channels first.
        kernel_size: The side-length of the sliding window used in comparison. Must be an odd value.
        kernel_sigma: Sigma of normal distribution.
        data_range: Value range of input images (usually 1.0 or 255).
        size_average: If size_average=True, ssim of all images will be averaged as a scalar.
        full: Return sc or not.
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        Value of Structural Similarity (SSIM) index. In case of 5D input tensors, complex value is returned
        as a tensor of size 2.
    References:
        .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
           (2004). Image quality assessment: From error visibility to
           structural similarity. IEEE Transactions on Image Processing,
           13, 600-612.
           https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
           :DOI:`10.1109/TIP.2003.819861`
    """
    _validate_input(input_tensors=(x, y), allow_5d=True, kernel_size=kernel_size, scale_weights=None)
    x, y = _adjust_dimensions(input_tensors=(x, y))
    kernel = _fspecial_gauss_1d(kernel_size, kernel_sigma)
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)

    _compute_ssim = _ssim_complex if x.dim() == 5 else _ssim
    ssim_val, cs = _compute_ssim(x=x, y=y, kernel=kernel, data_range=data_range, full=True, k1=k1, k2=k2)

    if size_average:
        ssim_val = ssim_val.mean(0)
        cs = cs.mean(0)

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
            \operatorname{mean}(1 - SSIM), &  \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(1 - SSIM),  &  \text{if reduction} = \text{'sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    The sum operation still operates over all the elements, and divides by :math:`n`.

    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

    In case of 5D input tensors, complex value is returned as a tensor of size 2.

    Args:
        kernel_size: By default, the mean and covariance of a pixel is obtained
            by convolution with given filter_size.
        kernel_sigma: Standard deviation for Gaussian kernel.
        k1: Coefficient related to c1 in the above equation.
        k2: Coefficient related to c2 in the above equation.
        reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        data_range: The difference between the maximum and minimum of the pixel value,
            i.e., if for image x it holds min(x) = 0 and max(x) = 1, then data_range = 1.
            The pixel value interval of both input and output should remain the same.

    Shape:
        - Input: Required to be 2D (H, W), 3D (C,H,W), 4D (N,C,H,W) or 5D (N,C,H,W,2), channels first.
        - Target: Required to be 2D (H, W), 3D (C,H,W), 4D (N,C,H,W) or 5D (N,C,H,W,2), channels first.

    Examples::

        >>> loss = SSIMLoss()
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
    __constants__ = ['filter_size', 'k1', 'k2', 'sigma', 'kernel', 'reduction']

    def __init__(self, kernel_size: int = 11, kernel_sigma: float = 1.5, k1: float = 0.01, k2: float = 0.03,
                 reduction: str = 'mean', data_range: Union[int, float] = 1.) -> None:
        super().__init__()

        # Generic loss parameters.
        self.reduction = reduction

        # Loss-specific parameters.
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self.k1 = k1
        self.k2 = k2
        self.data_range = data_range

        # Cash kernel between calls.
        self.kernel = _fspecial_gauss_1d(kernel_size, kernel_sigma)

    def forward(self,
                prediction: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        r"""Computation of Structural Similarity (SSIM) index as a loss function.

        Args:
            prediction: Tensor of prediction of the network.
            target: Reference tensor.

        Returns:
            Value of SSIM loss to be minimized. 0 <= SSIM loss <= 1. In case of 5D input tensors,
            complex value is returned as a tensor of size 2.
        """
        _validate_input(input_tensors=(prediction, target), allow_5d=True,
                        kernel_size=self.kernel_size, scale_weights=None)
        prediction, target = _adjust_dimensions(input_tensors=(prediction, target))

        return self.compute_metric(prediction, target)

    def compute_metric(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        kernel = self.kernel.repeat(prediction.shape[1], 1, 1, 1)
        kernel = kernel.to(device=prediction.device)

        _compute_ssim = _ssim_complex if prediction.dim() == 5 else _ssim
        ssim_val = _compute_ssim(
            x=prediction,
            y=target,
            kernel=kernel,
            data_range=self.data_range,
            full=False,
            k1=self.k1,
            k2=self.k2
        )
        ssim_loss = torch.ones_like(ssim_val) - ssim_val
        if self.reduction == 'mean':
            ssim_loss = torch.mean(ssim_loss, dim=0)
        elif self.reduction == 'sum':
            ssim_loss = torch.sum(ssim_loss, dim=0)
        elif self.reduction != 'none':
            raise ValueError(f'Expected reduction modes "mean"|"sum"|"none", got{self.reduction}')

        return ssim_loss


def multi_scale_ssim(x: torch.Tensor, y: torch.Tensor, kernel_size: int = 11, kernel_sigma: float = 1.5,
                     data_range: Union[int, float] = 255, size_average: bool = True,
                     scale_weights: Optional[Union[Tuple[float], List[float], torch.Tensor]] = None, k1=0.01,
                     k2=0.03) -> torch.Tensor:
    r""" Interface of Multi-scale Structural Similarity (MS-SSIM) index.
    Args:
        x: Batch of images. Required to be 2D (H, W), 3D (C,H,W), 4D (N,C,H,W) or 5D (N,C,H,W,2), channels first.
        y: Batch of images. Required to be 2D (H, W), 3D (C,H,W), 4D (N,C,H,W) or 5D (N,C,H,W,2), channels first.
        kernel_size: The side-length of the sliding window used in comparison. Must be an odd value.
        kernel_sigma: Sigma of normal distribution.
        data_range: Value range of input images (usually 1.0 or 255).
        size_average: If size_average=True, ssim of all images will be averaged as a scalar.
        scale_weights: Weights for different scales.
            If None, default weights from the paper [1] will be used.
            Default weights: (0.0448, 0.2856, 0.3001, 0.2363, 0.1333).
        k1: Algorithm parameter, K1 (small constant, see [2]).
        k2: Algorithm parameter, K2 (small constant, see [2]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        Value of Multi-scale Structural Similarity (MS-SSIM) index. In case of 5D input tensors,
        complex value is returned as a tensor of size 2.
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
    _validate_input(input_tensors=(x, y), allow_5d=True, kernel_size=kernel_size, scale_weights=scale_weights)
    x, y = _adjust_dimensions(input_tensors=(x, y))

    if scale_weights is None:
        scale_weights_from_ms_ssim_paper = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        scale_weights = scale_weights_from_ms_ssim_paper

    scale_weights_tensor = torch.tensor(scale_weights).to(x.device, dtype=x.dtype)
    kernel = _fspecial_gauss_1d(kernel_size, kernel_sigma)
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)

    _compute_msssim = _multi_scale_ssim_complex if x.dim() == 5 else _multi_scale_ssim
    msssim_val = _compute_msssim(
        x=x,
        y=y,
        data_range=data_range,
        kernel=kernel,
        scale_weights_tensor=scale_weights_tensor,
        k1=k1,
        k2=k2
    )

    if size_average:
        msssim_val = msssim_val.mean(0)

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
            \operatorname{mean}(1 - MSSIM), &  \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(1 - MSSIM),  &  \text{if reduction} = \text{'sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    The sum operation still operates over all the elements, and divides by :math:`n`.

    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

    In case of 5D input tensors, complex value is returned as a tensor of size 2.

   Args:
        kernel_size: By default, the mean and covariance of a pixel is obtained
            by convolution with given filter_size.
        kernel_sigma: Standard deviation for Gaussian kernel.
        k1: Coefficient related to c1 in the above equation.
        k2: Coefficient related to c2 in the above equation.
        scale_weights:  Weights for different scales.
            If None, default weights from the paper [1] will be used.
            Default weights: (0.0448, 0.2856, 0.3001, 0.2363, 0.1333).
        reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        data_range: The difference between the maximum and minimum of the pixel value,
            i.e., if for image x it holds min(x) = 0 and max(x) = 1, then data_range = 1.
            The pixel value interval of both input and output should remain the same.


    Shape:
        - Input: Required to be 2D (H, W), 3D (C,H,W), 4D (N,C,H,W) or 5D (N,C,H,W,2), channels first.
        - Target: Required to be 2D (H, W), 3D (C,H,W), 4D (N,C,H,W) or 5D (N,C,H,W,2), channels first.

    Examples::

        >>> loss = MultiScaleSSIMLoss()
        >>> input = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> target = torch.rand(3, 3, 256, 256)
        >>> output = loss(input, target)
        >>> output.backward()

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
    __constants__ = ['filter_size', 'k1', 'k2', 'sigma', 'kernel', 'reduction']

    def __init__(self, kernel_size: int = 11, kernel_sigma: float = 1.5, k1: float = 0.01, k2: float = 0.03,
                 scale_weights: Optional[Union[Tuple[float], List[float], torch.Tensor]] = None,
                 reduction: str = 'mean', data_range: Union[int, float] = 1.) -> None:
        super().__init__()

        # Generic loss parameters.
        self.reduction = reduction

        # Loss-specific parameters.
        if scale_weights is None:
            scale_weights_from_ms_ssim_paper = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
            scale_weights = scale_weights_from_ms_ssim_paper
        self.scale_weights = torch.tensor(scale_weights)
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
            Value of MS-SSIM loss to be minimized. 0 <= MS-SSIM loss <= 1. In case of 5D tensor,
            complex value is returned as a tensor of size 2.
        """
        _validate_input(input_tensors=(prediction, target), allow_5d=True,
                        kernel_size=self.kernel_size, scale_weights=self.scale_weights)
        prediction, target = _adjust_dimensions(input_tensors=(prediction, target))

        score = self.compute_metric(prediction, target)
        return score

    def compute_metric(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel.repeat(prediction.shape[1], 1, 1, 1)
        scale_weights_tensor = self.scale_weights.to(device=prediction.device, dtype=prediction.dtype)

        _compute_msssim = _multi_scale_ssim_complex if prediction.dim() == 5 else _multi_scale_ssim
        msssim_val = _compute_msssim(
            x=prediction,
            y=target,
            data_range=self.data_range,
            kernel=kernel,
            scale_weights_tensor=scale_weights_tensor,
            k1=self.k1,
            k2=self.k2)

        msssim_loss = torch.ones_like(msssim_val) - msssim_val
        if self.reduction == 'mean':
            msssim_loss = torch.mean(msssim_loss, dim=0)
        elif self.reduction == 'sum':
            msssim_loss = torch.sum(msssim_loss, dim=0)
        elif self.reduction != 'none':
            raise ValueError(f'Expected reduction modes "mean"|"sum"|"none", got{self.reduction}')

        return msssim_loss


def _fspecial_gauss_1d(size: int, sigma: float) -> torch.Tensor:
    r""" Creates a 1-D gauss kernel.

    Args:
        size: The size of gauss kernel.
        sigma: Sigma of normal distribution.

    Returns:
        1D Gauss kernel.
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def _ssim_per_channel(x: torch.Tensor, y: torch.Tensor, kernel: torch.Tensor, data_range: Union[float, int] = 255,
                      k1: float = 0.01, k2: float = 0.03) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Calculate Structural Similarity (SSIM) index for X and Y per channel.

        Args:
            x: Batch of images, (N,C,H,W).
            y: Batch of images, (N,C,H,W).
            kernel: 1-D gauss kernel.
            data_range: Value range of input images (usually 1.0 or 255).
            k1: Algorithm parameter, K1 (small constant, see [1]).
            k2: Algorithm parameter, K2 (small constant, see [1]).
                Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.


    Returns:
        Full Value of Structural Similarity (SSIM) index.
    """

    if x.size(-1) < kernel.size(-1) or x.size(-2) < kernel.size(-1):
        raise ValueError(f'Kernel size can\'t be greater than actual input size. Input size: {x.size()}. '
                         f'Kernel size: {kernel.size()}')

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

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

    ssim_val = ssim_map.mean(dim=(-1, -2))
    cs = cs_map.mean(dim=(-1, -2))

    return ssim_val, cs


def _ssim(x: torch.Tensor, y: torch.Tensor, kernel: torch.Tensor, data_range: Union[float, int] = 255,
          full: bool = False, k1: float = 0.01, k2: float = 0.03) \
        -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Calculate Structural Similarity (SSIM) index for X and Y.

    Args:
        x: Batch of images, (N,C,H,W).
        y: Batch of images, (N,C,H,W).
        kernel: 1-D gauss kernel.
        data_range: Value range of input images (usually 1.0 or 255).
        full: Return sc or not.
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Value of Structural Similarity (SSIM) index.
    """

    ssim_map, cs_map = _ssim_per_channel(x=x, y=y, kernel=kernel, data_range=data_range, k1=k1, k2=k2)

    ssim_val = ssim_map.mean(1)
    cs = cs_map.mean(1)

    if full:
        return ssim_val, cs

    return ssim_val


def _multi_scale_ssim(x: torch.Tensor, y: torch.Tensor, data_range: Union[int, float], kernel: torch.Tensor,
                      scale_weights_tensor: torch.Tensor, k1: float, k2: float) -> torch.Tensor:
    levels = scale_weights_tensor.size(0)
    min_size = (kernel.size(-1) - 1) * 2 ** (levels - 1) + 1
    if x.size(-1) < min_size or x.size(-2) < min_size:
        raise ValueError(f'Invalid size of the input images, expected at least {min_size}x{min_size}.')

    mcs = []
    ssim_val = None
    for _ in range(levels):
        ssim_val, cs = _ssim_per_channel(x, y, kernel=kernel, data_range=data_range, k1=k1, k2=k2)
        mcs.append(cs)

        padding = (x.shape[2] % 2, x.shape[3] % 2)
        x = f.avg_pool2d(x, kernel_size=2, padding=padding)
        y = f.avg_pool2d(y, kernel_size=2, padding=padding)

    # mcs, (level, batch)
    mcs_ssim = torch.relu(torch.stack(mcs[:-1] + [ssim_val], dim=0))

    # weights, (level)
    msssim_val = torch.prod((mcs_ssim ** scale_weights_tensor.view(-1, 1, 1)), dim=0).mean(1)

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


def _ssim_per_channel_complex(x: torch.Tensor, y: torch.Tensor, kernel: torch.Tensor,
                              data_range: Union[float, int] = 255, k1: float = 0.01,
                              k2: float = 0.03) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Calculate Structural Similarity (SSIM) index for Complex X and Y per channel.

        Args:
            x: Batch of complex images, (N,C,H,W,2).
            y: Batch of complex images, (N,C,H,W,2).
            kernel: 1-D gauss kernel.
            data_range: Value range of input images (usually 1.0 or 255).
            k1: Algorithm parameter, K1 (small constant, see [1]).
            k2: Algorithm parameter, K2 (small constant, see [1]).
                Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.


    Returns:
        Full Value of Complex Structural Similarity (SSIM) index.
    """

    if x.size(-2) < kernel.size(-1) or x.size(-3) < kernel.size(-1):
        raise ValueError(f'Kernel size can\'t be greater than actual input size. Input size: {x.size()}. '
                         f'Kernel size: {kernel.size()}')

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    kernel = kernel.to(x.device, dtype=x.dtype)

    x_real = x[..., 0]
    x_imag = x[..., 1]
    y_real = y[..., 0]
    y_imag = y[..., 1]

    mu1_real = _gaussian_filter(x_real, kernel)
    mu1_imag = _gaussian_filter(x_imag, kernel)
    mu2_real = _gaussian_filter(y_real, kernel)
    mu2_imag = _gaussian_filter(y_imag, kernel)

    mu1_sq = mu1_real.pow(2) + mu1_imag.pow(2)
    mu2_sq = mu2_real.pow(2) + mu2_imag.pow(2)
    mu1_mu2_real = mu1_real * mu2_real - mu1_imag * mu2_imag
    mu1_mu2_imag = mu1_real * mu2_imag + mu1_imag * mu2_real

    compensation = 1.0

    x_sq = x_real.pow(2) + x_imag.pow(2)
    y_sq = y_real.pow(2) + y_imag.pow(2)
    x_y_real = x_real * y_real - x_imag * y_imag
    x_y_imag = x_real * y_imag + x_imag * y_real

    sigma1_sq = compensation * _gaussian_filter(x_sq, kernel) - mu1_sq
    sigma2_sq = compensation * _gaussian_filter(y_sq, kernel) - mu2_sq
    sigma12_real = compensation * _gaussian_filter(x_y_real, kernel) - mu1_mu2_real
    sigma12_imag = compensation * _gaussian_filter(x_y_imag, kernel) - mu1_mu2_imag
    sigma12 = torch.stack((sigma12_imag, sigma12_real), dim=-1)
    mu1_mu2 = torch.stack((mu1_mu2_real, mu1_mu2_imag), dim=-1)
    # Set alpha = beta = gamma = 1.
    cs_map = (sigma12 * 2 + c2) / (sigma1_sq.unsqueeze(-1) + sigma2_sq.unsqueeze(-1) + c2)
    ssim_map = ((mu1_mu2 * 2 + c1) / (mu1_sq.unsqueeze(-1) + mu2_sq.unsqueeze(-1) + c1)) * cs_map

    ssim_val = ssim_map.mean(dim=(-2, -3))
    cs = cs_map.mean(dim=(-2, -3))

    return ssim_val, cs


def _ssim_complex(x: torch.Tensor, y: torch.Tensor, kernel: torch.Tensor, data_range: Union[float, int] = 255,
                  full: bool = False, k1: float = 0.01, k2: float = 0.03) \
        -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Calculate Structural Similarity (SSIM) index for Complex X and Y.

    Args:
        x: Batch of complex images, (N,C,H,W,2).
        y: Batch of complex images, (N,C,H,W,2).
        kernel: 1-D gauss kernel.
        data_range: Value range of input images (usually 1.0 or 255).
        full: Return sc or not.
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Value of Complex Structural Similarity (SSIM) index.
    """
    ssim_map, cs_map = _ssim_per_channel_complex(x=x, y=y, kernel=kernel, data_range=data_range, k1=k1, k2=k2)

    ssim_val = ssim_map.mean(1)
    cs = cs_map.mean(1)

    if full:
        return ssim_val, cs

    return ssim_val


def _multi_scale_ssim_complex(x: torch.Tensor, y: torch.Tensor, data_range: Union[int, float],
                              kernel: torch.Tensor, scale_weights_tensor: torch.Tensor, k1: float,
                              k2: float) -> torch.Tensor:
    levels = scale_weights_tensor.size(0)
    min_size = (kernel.size(-1) - 1) * 2 ** (levels - 1) + 1
    if x.size(-2) < min_size or x.size(-3) < min_size:
        raise ValueError(f'Invalid size of the input images, expected at least {min_size}x{min_size}.')

    mcs = []
    ssim_val = None
    for _ in range(levels):
        ssim_val, cs = _ssim_per_channel_complex(x, y, kernel=kernel, data_range=data_range, k1=k1, k2=k2)

        x_real = x[..., 0]
        x_imag = x[..., 1]
        y_real = y[..., 0]
        y_imag = y[..., 1]
        mcs.append(cs)

        padding = (x.size(2) % 2, x.size(3) % 2)
        x_real = f.avg_pool2d(x_real, kernel_size=2, padding=padding)
        x_imag = f.avg_pool2d(x_imag, kernel_size=2, padding=padding)
        y_real = f.avg_pool2d(y_real, kernel_size=2, padding=padding)
        y_imag = f.avg_pool2d(y_imag, kernel_size=2, padding=padding)
        x = torch.stack((x_real, x_imag), dim=-1)
        y = torch.stack((y_real, y_imag), dim=-1)

    # mcs, (level, batch)
    mcs_ssim = torch.relu(torch.stack(mcs[:-1] + [ssim_val], dim=0))

    mcs_ssim_real = mcs_ssim[..., 0]
    mcs_ssim_imag = mcs_ssim[..., 1]
    mcs_ssim_abs = (mcs_ssim_real.pow(2) + mcs_ssim_imag.pow(2)).sqrt()
    mcs_ssim_deg = torch.atan(mcs_ssim_imag / mcs_ssim_real)

    mcs_ssim_pow_abs = mcs_ssim_abs ** scale_weights_tensor.view(-1, 1, 1)
    mcs_ssim_pow_deg = mcs_ssim_deg * scale_weights_tensor.view(-1, 1, 1)

    msssim_val_abs = torch.prod(mcs_ssim_pow_abs, dim=0)
    msssim_val_deg = torch.sum(mcs_ssim_pow_deg, dim=0)
    msssim_val_real = msssim_val_abs * torch.cos(msssim_val_deg)
    msssim_val_imag = msssim_val_abs * torch.sin(msssim_val_deg)
    msssim_val = torch.stack((msssim_val_real, msssim_val_imag), dim=-1).mean(dim=1)

    return msssim_val
