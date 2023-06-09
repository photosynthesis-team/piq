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
from typing import Optional, Union

import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from piq.utils import _validate_input, _reduce
from piq.functional import similarity_map, gradient_map, prewitt_filter, rgb2yiq


def gmsd(x: torch.Tensor, y: torch.Tensor, reduction: str = 'mean',
         data_range: Union[int, float] = 1., t: float = 170 / (255. ** 2)) -> torch.Tensor:
    r"""Compute Gradient Magnitude Similarity Deviation.

    Supports greyscale and colour images with RGB channel order.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        data_range: Maximum value range of images (usually 1.0 or 255).
        t: Constant from the reference paper numerical stability of similarity map.

    Returns:
        Gradient Magnitude Similarity Deviation between given tensors.

    References:
        Wufeng Xue et al. Gradient Magnitude Similarity Deviation (2013)
        https://arxiv.org/pdf/1308.3052.pdf
    """
    _validate_input([x, y], dim_range=(4, 4), data_range=(0, data_range))

    # Rescale
    x = x / float(data_range)
    y = y / float(data_range)

    num_channels = x.size(1)
    if num_channels == 3:
        x = rgb2yiq(x)[:, :1]
        y = rgb2yiq(y)[:, :1]
    up_pad = 0
    down_pad = max(x.shape[2] % 2, x.shape[3] % 2)
    pad_to_use = [up_pad, down_pad, up_pad, down_pad]
    x = F.pad(x, pad=pad_to_use)
    y = F.pad(y, pad=pad_to_use)

    x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
    y = F.avg_pool2d(y, kernel_size=2, stride=2, padding=0)

    score = _gmsd(x=x, y=y, t=t)
    return _reduce(score, reduction)


def _gmsd(x: torch.Tensor, y: torch.Tensor,
          t: float = 170 / (255. ** 2), alpha: float = 0.0) -> torch.Tensor:
    r"""Compute Gradient Magnitude Similarity Deviation
    Supports greyscale images in [0, 1] range.

    Args:
        x: Tensor. Shape :math:`(N, 1, H, W)`.
        y: Tensor. Shape :math:`(N, 1, H, W)`.
        t: Constant from the reference paper numerical stability of similarity map
        alpha: Masking coefficient for similarity masks computation

    Returns:
        gmsd : Gradient Magnitude Similarity Deviation between given tensors.

    References:
        https://arxiv.org/pdf/1308.3052.pdf
    """

    # Compute grad direction
    p_filter = prewitt_filter(dtype=x.dtype, device=x.device)
    kernels = torch.stack([p_filter, p_filter.transpose(-1, -2)])
    x_grad = gradient_map(x, kernels)
    y_grad = gradient_map(y, kernels)

    # Compute GMS
    gms = similarity_map(x_grad, y_grad, constant=t, alpha=alpha)
    mean_gms = torch.mean(gms, dim=[1, 2, 3], keepdims=True)

    # Compute GMSD along spatial dimensions. Shape (batch_size )
    score = torch.pow(gms - mean_gms, 2).mean(dim=[1, 2, 3]).sqrt()
    return score


class GMSDLoss(_Loss):
    r"""Creates a criterion that measures Gradient Magnitude Similarity Deviation
    between each element in the input and target.

    Args:
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        data_range: Maximum value range of images (usually 1.0 or 255).
        t: Constant from the reference paper numerical stability of similarity map

    Examples:
        >>> loss = GMSDLoss()
        >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> y = torch.rand(3, 3, 256, 256)
        >>> output = loss(x, y)
        >>> output.backward()

    References:
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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""Computation of Gradient Magnitude Similarity Deviation (GMSD) as a loss function.
        Supports greyscale and colour images with RGB channel order.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Value of GMSD loss to be minimized in [0, 1] range.
        """
        return gmsd(x=x, y=y, reduction=self.reduction, data_range=self.data_range, t=self.t)


def multi_scale_gmsd(x: torch.Tensor, y: torch.Tensor, data_range: Union[int, float] = 1., reduction: str = 'mean',
                     scale_weights: Optional[torch.Tensor] = None,
                     chromatic: bool = False, alpha: float = 0.5, beta1: float = 0.01, beta2: float = 0.32,
                     beta3: float = 15., t: float = 170) -> torch.Tensor:
    r"""Computation of Multi scale GMSD.

    Supports greyscale and colour images with RGB channel order.
    The height and width should be at least ``2 ** scales + 1``.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        data_range: Maximum value range of images (usually 1.0 or 255).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        scale_weights: Weights for different scales. Can contain any number of floating point values.
        chromatic: Flag to use MS-GMSDc algorithm from paper.
            It also evaluates chromatic components of the image. Default: True
        alpha: Masking coefficient. See references for details.
        beta1: Algorithm parameter. Weight of chromatic component in the loss.
        beta2: Algorithm parameter. Small constant, see references.
        beta3: Algorithm parameter. Small constant, see references.
        t: Constant from the reference paper numerical stability of similarity map

    Returns:
        Value of MS-GMSD in [0, 1] range.

    References:
        Bo Zhang et al. Gradient Magnitude Similarity Deviation on Multiple Scales (2017).
        http://www.cse.ust.hk/~psander/docs/gradsim.pdf
    """
    _validate_input([x, y], dim_range=(4, 4), data_range=(0, data_range))

    # Rescale
    x = x / float(data_range) * 255
    y = y / float(data_range) * 255

    # Values from the paper
    if scale_weights is None:
        scale_weights = torch.tensor([0.096, 0.596, 0.289, 0.019], device=x.device, dtype=x.dtype)
    else:
        # Normalize scale weights
        scale_weights = scale_weights / scale_weights.sum()

    # Check that input is big enough
    num_scales = scale_weights.size(0)
    min_size = 2 ** num_scales + 1

    if x.size(-1) < min_size or x.size(-2) < min_size:
        raise ValueError(f'Invalid size of the input images, expected at least {min_size}x{min_size}.')

    num_channels = x.size(1)
    if num_channels == 3:
        x = rgb2yiq(x)
        y = rgb2yiq(y)

    ms_gmds = []
    for scale in range(num_scales):
        if scale > 0:

            # Average by 2x2 filter and downsample
            up_pad = 0
            down_pad = max(x.shape[2] % 2, x.shape[3] % 2)
            pad_to_use = [up_pad, down_pad, up_pad, down_pad]
            x = F.pad(x, pad=pad_to_use)
            y = F.pad(y, pad=pad_to_use)
            x = F.avg_pool2d(x, kernel_size=2, padding=0)
            y = F.avg_pool2d(y, kernel_size=2, padding=0)

        score = _gmsd(x[:, :1], y[:, :1], t=t, alpha=alpha)
        ms_gmds.append(score)

    # Stack results in different scales and multiply by weight
    ms_gmds_val = scale_weights.view(1, num_scales) * (torch.stack(ms_gmds, dim=1) ** 2)

    # Sum and take sqrt per-image
    ms_gmds_val = torch.sqrt(torch.sum(ms_gmds_val, dim=1))

    # Shape: (batch_size, )
    score = ms_gmds_val

    if chromatic:
        assert x.size(1) == 3, "Chromatic component can be computed only for RGB images!"

        x_iq = x[:, 1:]
        y_iq = y[:, 1:]

        rmse_iq = torch.sqrt(torch.mean((x_iq - y_iq) ** 2, dim=[2, 3]))
        rmse_chrome = torch.sqrt(torch.sum(rmse_iq ** 2, dim=1))
        gamma = 2 / (1 + beta2 * torch.exp(-beta3 * ms_gmds_val)) - 1

        score = gamma * ms_gmds_val + (1 - gamma) * beta1 * rmse_chrome

    return _reduce(score, reduction)


class MultiScaleGMSDLoss(_Loss):
    r"""Creates a criterion that measures multi scale Gradient Magnitude Similarity Deviation
    between each element in the input :math:`x` and target :math:`y`.

    Args:
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        data_range: Maximum value range of images (usually 1.0 or 255).
        scale_weights: Weights for different scales. Can contain any number of floating point values.
            By default weights are initialized with values from the paper.
        chromatic: Flag to use MS-GMSDc algorithm from paper.
            It also evaluates chromatic components of the image. Default: True
        beta1: Algorithm parameter. Weight of chromatic component in the loss.
        beta2: Algorithm parameter. Small constant, references.
        beta3: Algorithm parameter. Small constant, references.
        t: Constant from the reference paper numerical stability of similarity map

    Examples:
        >>> loss = MultiScaleGMSDLoss()
        >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> y = torch.rand(3, 3, 256, 256)
        >>> output = loss(x, y)
        >>> output.backward()

    References:
        Bo Zhang et al. Gradient Magnitude Similarity Deviation on Multiple Scales (2017).
        http://www.cse.ust.hk/~psander/docs/gradsim.pdf
    """

    def __init__(self, reduction: str = 'mean', data_range: Union[int, float] = 1.,
                 scale_weights: Optional[torch.Tensor] = None,
                 chromatic: bool = False, alpha: float = 0.5, beta1: float = 0.01, beta2: float = 0.32,
                 beta3: float = 15., t: float = 170) -> None:
        super().__init__()

        # Generic loss parameters.
        self.reduction = reduction

        # Loss-specific parameters.
        self.data_range = data_range

        if scale_weights is None:
            self.register_buffer("scale_weights", torch.tensor([0.096, 0.596, 0.289, 0.019]))
        else:
            self.register_buffer("scale_weights", scale_weights)

        self.chromatic = chromatic
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.t = t

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""Computation of Multi Scale GMSD index as a loss function.
        Supports greyscale and colour images with RGB channel order.
        The height and width should be at least 2 ** scales + 1.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Value of MS-GMSD loss to be minimized in [0, 1] range.
        """
        return multi_scale_gmsd(x=x, y=y, data_range=self.data_range,
                                reduction=self.reduction, chromatic=self.chromatic, alpha=self.alpha, beta1=self.beta1,
                                beta2=self.beta2, beta3=self.beta3, scale_weights=self.scale_weights, t=self.t)
