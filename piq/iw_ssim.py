r""" Implementation of Information Content Weighted Structural Similarity Index (IW-SSIM).

Information Content Weighted Structural Similarity Index (IW-SSIM) [1] is an extension of
the structural similarity (SSIM). IW-SSIM uses the idea of information content weighted pooling for similarity
evaluation.

Estimation values produced by presented implementation corresponds to MATLAB based estimations [2].

References:
  [1] Wang, Zhou, and Qiang Li.
    "Information content weighting for perceptual image quality assessment."
    IEEE Transactions on image processing 20.5 (2011): 1185-1198.
    https://ece.uwaterloo.ca/~z70wang/publications/IWSSIM.pdf

  [2] https://ece.uwaterloo.ca/~z70wang/research/iwssim/
"""

import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from piq.utils import _validate_input, _reduce, _parse_version
from piq.functional import gaussian_filter, binomial_filter1d, average_filter2d, rgb2yiq
from typing import Union, Optional, Tuple
import math


def information_weighted_ssim(x: torch.Tensor, y: torch.Tensor, data_range: Union[int, float] = 1.,
                              kernel_size: int = 11, kernel_sigma: float = 1.5, k1: float = 0.01, k2: float = 0.03,
                              parent: bool = True, blk_size: int = 3, sigma_nsq: float = 0.4,
                              scale_weights: Optional[torch.Tensor] = None,
                              reduction: str = 'mean') -> torch.Tensor:
    r"""Interface of Information Content Weighted Structural Similarity (IW-SSIM) index.
    Inputs supposed to be in range ``[0, data_range]``.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        data_range: Maximum value range of images (usually 1.0 or 255).
        kernel_size: The side-length of the sliding window used in comparison. Must be an odd value.
        kernel_sigma: Sigma of normal distribution for sliding window used in comparison.
        k1: Algorithm parameter, K1 (small constant).
        k2: Algorithm parameter, K2 (small constant).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        parent: Flag to control dependency on previous layer of pyramid.
        blk_size: The side-length of the sliding window used in comparison for information content.
        sigma_nsq: Parameter of visual distortion model.
        scale_weights: Weights for scaling.
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``

    Returns:
        Value of Information Content Weighted Structural Similarity (IW-SSIM) index.

    References:
        Wang, Zhou, and Qiang Li..
        Information content weighting for perceptual image quality assessment.
        IEEE Transactions on image processing 20.5 (2011): 1185-1198.
        https://ece.uwaterloo.ca/~z70wang/publications/IWSSIM.pdf DOI:`10.1109/TIP.2010.2092435`

    Note:
        Lack of content in target image could lead to RuntimeError due to singular information content matrix,
        which cannot be inverted.
    """
    assert kernel_size % 2 == 1, f'Kernel size must be odd, got [{kernel_size}]'

    _validate_input(tensors=[x, y], dim_range=(4, 4), data_range=(0., data_range))

    x = x / float(data_range) * 255
    y = y / float(data_range) * 255

    if x.size(1) == 3:
        x = rgb2yiq(x)[:, :1]
        y = rgb2yiq(y)[:, :1]

    if scale_weights is None:
        scale_weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=x.dtype, device=x.device)
    scale_weights = scale_weights / scale_weights.sum()
    if scale_weights.size(0) != scale_weights.numel():
        raise ValueError(f'Expected a vector of weights, got {scale_weights.dim()}D tensor')

    levels = scale_weights.size(0)

    min_size = (kernel_size - 1) * 2 ** (levels - 1) + 1
    if x.size(-1) < min_size or x.size(-2) < min_size:
        raise ValueError(f'Invalid size of the input images, expected at least {min_size}x{min_size}.')

    blur_pad = math.ceil((kernel_size - 1) / 2)  # Ceil
    iw_pad = blur_pad - math.floor((blk_size - 1) / 2)  # floor
    gauss_kernel = gaussian_filter(kernel_size, kernel_sigma, device=x.device, dtype=x.dtype).repeat(x.size(1), 1, 1, 1)

    # Size of the kernel size to build Laplacian pyramid
    pyramid_kernel_size = 5
    bin_filter = binomial_filter1d(kernel_size=pyramid_kernel_size, device=x.device, dtype=x.dtype) * 2 ** 0.5

    lo_x, x_diff_old = _pyr_step(x, bin_filter)
    lo_y, y_diff_old = _pyr_step(y, bin_filter)

    x = lo_x
    y = lo_y
    wmcs = []

    for i in range(levels):
        if i < levels - 2:
            lo_x, x_diff = _pyr_step(x, bin_filter)
            lo_y, y_diff = _pyr_step(y, bin_filter)
            x = lo_x
            y = lo_y

        else:
            x_diff = x
            y_diff = y

        ssim_map, cs_map = _ssim_per_channel(x=x_diff_old, y=y_diff_old, kernel=gauss_kernel, data_range=255,
                                             k1=k1, k2=k2)

        if parent and i < levels - 2:
            iw_map = _information_content(x=x_diff_old, y=y_diff_old, y_parent=y_diff, kernel_size=blk_size,
                                          sigma_nsq=sigma_nsq)

            iw_map = iw_map[:, :, iw_pad:-iw_pad, iw_pad:-iw_pad]

        elif i == levels - 1:
            iw_map = torch.ones_like(cs_map)
            cs_map = ssim_map

        else:
            iw_map = _information_content(x=x_diff_old, y=y_diff_old, y_parent=None, kernel_size=blk_size,
                                          sigma_nsq=sigma_nsq)
            iw_map = iw_map[:, :, iw_pad:-iw_pad, iw_pad:-iw_pad]

        wmcs.append(torch.sum(cs_map * iw_map, dim=(-2, -1)) / torch.sum(iw_map, dim=(-2, -1)))

        x_diff_old = x_diff
        y_diff_old = y_diff

    wmcs = torch.stack(wmcs, dim=0).abs()

    score = torch.prod((wmcs ** scale_weights.view(-1, 1, 1)), dim=0)[:, 0]

    return _reduce(x=score, reduction=reduction)


class InformationWeightedSSIMLoss(_Loss):
    r"""Creates a criterion that measures the Interface of Information Content Weighted Structural Similarity (IW-SSIM)
    index error betweeneach element in the input :math:`x` and target :math:`y`.

    Inputs supposed to be in range ``[0, data_range]``.

    If :attr:`reduction` is not ``'none'`` (default ``'mean'``), then:

    .. math::
        InformationWeightedSSIMLoss(x, y) =
        \begin{cases}
            \operatorname{mean}(1 - IWSSIM), &  \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(1 - IWSSIM),  &  \text{if reduction} = \text{'sum'.}
        \end{cases}

    Args:
        data_range: Maximum value range of images (usually 1.0 or 255).
        kernel_size: The side-length of the sliding window used in comparison. Must be an odd value.
        kernel_sigma: Sigma of normal distribution for sliding window used in comparison.
        k1: Algorithm parameter, K1 (small constant).
        k2: Algorithm parameter, K2 (small constant).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        parent: Flag to control dependency on previous layer of pyramid.
        blk_size: The side-length of the sliding window used in comparison for information content.
        sigma_nsq: Sigma of normal distribution for sliding window used in comparison for information content.
        scale_weights: Weights for scaling.
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``

    Examples:
        >>> loss = InformationWeightedSSIMLoss()
        >>> input = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> target = torch.rand(3, 3, 256, 256)
        >>> output = loss(input, target)
        >>> output.backward()

    References:
        Wang, Zhou, and Qiang Li..
        Information content weighting for perceptual image quality assessment.
        IEEE Transactions on image processing 20.5 (2011): 1185-1198.
        https://ece.uwaterloo.ca/~z70wang/publications/IWSSIM.pdf DOI:`10.1109/TIP.2010.2092435`

    """

    def __init__(self, data_range: Union[int, float] = 1., kernel_size: int = 11, kernel_sigma: float = 1.5,
                 k1: float = 0.01, k2: float = 0.03, parent: bool = True, blk_size: int = 3, sigma_nsq: float = 0.4,
                 scale_weights: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.data_range = data_range
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self.k1 = k1
        self.k2 = k2
        self.parent = parent
        self.blk_size = blk_size
        self.sigma_nsq = sigma_nsq
        self.scale_weights = scale_weights
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""Computation of Information Content Weighted Structural Similarity (IW-SSIM) index as a loss function.
        For colour images channel order is RGB.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Value of IW-SSIM loss to be minimized, i.e. ``1 - information_weighted_ssim`` in [0, 1] range.
        """
        score = information_weighted_ssim(x=x, y=y, data_range=self.data_range, kernel_size=self.kernel_size,
                                          kernel_sigma=self.kernel_sigma, k1=self.k1, k2=self.k2,
                                          parent=self.parent, blk_size=self.blk_size, sigma_nsq=self.sigma_nsq,
                                          scale_weights=self.scale_weights, reduction=self.reduction)
        return torch.ones_like(score) - score


def _pyr_step(x: torch.Tensor, kernel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r""" Computes one step of Laplacian pyramid generation.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        kernel: Kernel to perform blurring.

    Returns:
        Tuple of tensors with downscaled low resolution image and high-resolution difference.
    """
    # Blur and Downsampling
    up_pad = (kernel.size(-1) - 1) // 2  # 5 -> 2, 4 -> 1
    down_pad = kernel.size(-1) - 1 - up_pad  # 5 -> 2, 4 -> 2
    kernel_t = kernel.transpose(-2, -1)
    lo_x = x
    if x.size(-1) > 1:
        lo_x = F.pad(lo_x, pad=[up_pad, down_pad, 0, 0], mode='reflect')
        lo_x = F.conv2d(input=lo_x, weight=kernel.unsqueeze(0), padding=0)[:, :, :, ::2]
    if x.size(-2) > 1:
        lo_x = F.pad(lo_x, pad=[0, 0, up_pad, down_pad], mode='reflect')
        lo_x = F.conv2d(input=lo_x, weight=kernel_t.unsqueeze(0), padding=0)[:, :, ::2, :]

    # Upsampling and Blur
    up_pad = (kernel.size(-1) - 1) // 2  # 5 -> 2, 4 -> 1
    down_pad = kernel.size(-1) - 1 - up_pad  # 5 -> 2, 4 -> 2
    hi_x = lo_x

    if x.size(-1) > 1:
        upsampling_kernel = torch.tensor([[[[1., 0.]]]], dtype=x.dtype, device=x.device)
        hi_x = F.conv_transpose2d(input=hi_x, weight=upsampling_kernel, stride=(1, 2), padding=0)
        hi_x = F.pad(hi_x, pad=[up_pad, down_pad, 0, 0], mode='reflect')
        hi_x = F.conv2d(input=hi_x, weight=kernel.unsqueeze(0), padding=0)[:, :, :, :x.size(-1)]
    if x.size(-2) > 1:
        upsampling_kernel = torch.tensor([[[[1.], [0.]]]], dtype=x.dtype, device=x.device)
        hi_x = F.conv_transpose2d(input=hi_x, weight=upsampling_kernel, stride=(2, 1), padding=0)
        hi_x = F.pad(hi_x, pad=[0, 0, up_pad, down_pad], mode='reflect')
        hi_x = F.conv2d(input=hi_x, weight=kernel_t.unsqueeze(0), padding=0)[:, :, :x.size(-2), :]

    hi_x = x - hi_x
    return lo_x, hi_x


def _ssim_per_channel(x: torch.Tensor, y: torch.Tensor, kernel: torch.Tensor,
                      data_range: Union[float, int] = 1., k1: float = 0.01,
                      k2: float = 0.03) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Computes Structural Similarity (SSIM) index for X and Y per channel.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        kernel: 2D Gaussian kernel.
        data_range: Maximum value range of images (usually 1.0 or 255).
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Tuple with Structural Similarity maps and Contrast maps.
    """
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    n_channels = x.size(1)
    mu_x = F.conv2d(x, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu_y = F.conv2d(y, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu_xx = mu_x ** 2
    mu_yy = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_xx = F.conv2d(x ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xx
    sigma_yy = F.conv2d(y ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_yy
    sigma_xy = F.conv2d(x * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xy

    sigma_xx = F.relu(sigma_xx)
    sigma_yy = F.relu(sigma_yy)

    # Contrast sensitivity (CS) with alpha = beta = gamma = 1.
    cs_map = (2. * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)

    # Structural similarity (SSIM)
    ss_map = (2. * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs_map

    return ss_map, cs_map


def _information_content(x: torch.Tensor, y: torch.Tensor, y_parent: torch.Tensor = None,
                         kernel_size: int = 3, sigma_nsq: float = 0.4) -> torch.Tensor:
    r"""Computes Information Content Map for weighting the Structural Similarity.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        y_parent: Flag to control dependency on previous layer of pyramid.
        kernel_size: The side-length of the sliding window used in comparison for information content.
        sigma_nsq: Parameter of visual distortion model.

    Returns:
        Information Content Maps.
    """

    EPS = torch.finfo(x.dtype).eps
    n_channels = x.size(1)
    kernel = average_filter2d(kernel_size=kernel_size, device=x.device, dtype=x.dtype).repeat(x.size(1), 1, 1, 1)
    padding_up = kernel.size(-1) // 2
    padding_down = kernel.size(-1) - padding_up

    mu_x = F.conv2d(input=F.pad(x, pad=[padding_up, padding_down, padding_up, padding_down]), weight=kernel, padding=0,
                    groups=n_channels)
    mu_y = F.conv2d(input=F.pad(y, pad=[padding_up, padding_down, padding_up, padding_down]), weight=kernel, padding=0,
                    groups=n_channels)

    mu_xx = mu_x ** 2
    mu_yy = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_xx = F.conv2d(F.pad(x ** 2, pad=[padding_up, padding_down, padding_up, padding_down]), weight=kernel,
                        stride=1, padding=0, groups=n_channels) - mu_xx
    sigma_yy = F.conv2d(F.pad(y ** 2, pad=[padding_up, padding_down, padding_up, padding_down]), weight=kernel,
                        stride=1, padding=0, groups=n_channels) - mu_yy
    sigma_xy = F.conv2d(F.pad(x * y, pad=[padding_up, padding_down, padding_up, padding_down]), weight=kernel,
                        stride=1, padding=0, groups=n_channels) - mu_xy

    sigma_xx = F.relu(sigma_xx)
    sigma_yy = F.relu(sigma_yy)

    g = sigma_xy / (sigma_yy + EPS)
    vv = sigma_xx - g * sigma_xy
    g = g.masked_fill(sigma_yy < EPS, 0)
    vv[sigma_yy < EPS] = sigma_xx[sigma_yy < EPS]
    g = g.masked_fill(sigma_xx < EPS, 0)
    vv = vv.masked_fill(sigma_xx < EPS, 0)

    block = [kernel_size, kernel_size]

    nblv = y.size(-2) - block[0] + 1
    nblh = y.size(-1) - block[1] + 1
    nexp = nblv * nblh
    N = block[0] * block[1]

    assert block[0] % 2 == 1 and block[1] % 2 == 1, f'Expected odd block dimensions, got {block}'

    Ly = (block[0] - 1) // 2
    Lx = (block[1] - 1) // 2

    if y_parent is not None:
        # upscale y_parent and cut to the size of y

        y_parent_up = _image_enlarge(y_parent)[:, :, :y.size(-2), :y.size(-1)]
        N = N + 1

    Y = torch.zeros(y.size(0), y.size(1), nexp, N, dtype=y.dtype, device=y.device)

    n = -1
    for ny in range(-Ly, Ly + 1):
        for nx in range(-Lx, Lx + 1):
            n = n + 1
            foo = _shift(y, [ny, nx])
            foo = foo[:, :, Ly:Ly + nblv, Lx:Lx + nblh]
            Y[..., n] = foo.flatten(start_dim=-2, end_dim=-1)

    if y_parent is not None:
        n = n + 1
        foo = y_parent_up
        foo = foo[:, :, Ly:Ly + nblv, Lx:Lx + nblh]
        Y[..., n] = foo.flatten(start_dim=-2, end_dim=-1)

    C_u = torch.matmul(Y.transpose(-2, -1), Y) / nexp

    recommended_torch_version = _parse_version('1.10.0')
    torch_version = _parse_version(torch.__version__)
    if len(torch_version) != 0 and torch_version >= recommended_torch_version:
        eig_values, eig_vectors = torch.linalg.eigh(C_u)
    else:
        eig_values, eig_vectors = torch.symeig(C_u, eigenvectors=True)

    sum_eig_values = torch.sum(eig_values, dim=-1).view(y.size(0), y.size(1), 1, 1)
    non_zero_eig_values_matrix = torch.diag_embed(eig_values * (eig_values > 0))
    sum_non_zero_eig_values = torch.sum(non_zero_eig_values_matrix, dim=(-2, -1), keepdim=True)

    L = non_zero_eig_values_matrix * sum_eig_values / (sum_non_zero_eig_values + (sum_non_zero_eig_values == 0))

    C_u = torch.matmul(torch.matmul(eig_vectors, L), eig_vectors.transpose(-2, -1))

    C_u_inv = torch.inverse(C_u)

    ss = torch.matmul(Y, C_u_inv) * Y / N
    ss = torch.sum(ss, dim=-1, keepdim=True)
    ss = ss.view(y.size(0), y.size(1), nblv, nblh)
    g = g[:, :, Ly: Ly + nblv, Lx: Lx + nblh]
    vv = vv[:, :, Ly: Ly + nblv, Lx: Lx + nblh]

    # Calculate mutual information
    scaled_eig_values = torch.diagonal(L, offset=0, dim1=-2, dim2=-1).unsqueeze(2).unsqueeze(3)

    iw_map = torch.sum(torch.log2(1 + ((vv.unsqueeze(-1) + (1 + g.unsqueeze(-1) * g.unsqueeze(-1)) * sigma_nsq)
                                       * ss.unsqueeze(-1) * scaled_eig_values + sigma_nsq * vv.unsqueeze(-1)) / (
                                              sigma_nsq * sigma_nsq)), dim=-1)

    iw_map[iw_map < EPS] = 0

    return iw_map


def _image_enlarge(x: torch.Tensor) -> torch.Tensor:
    r"""Custom bilinear upscaling of an image.
    The function upscales an input image with upscaling factor 4x-3, adds padding on boundaries as difference
    and downscaled by the factor of 2.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.

    Returns:
        Upscaled tensor.
    """
    t1 = F.interpolate(x, size=(int(4 * x.size(-2) - 3), int(4 * x.size(-1) - 3)), mode='bilinear', align_corners=False)
    t2 = torch.zeros([x.size(0), 1, 4 * x.size(-2) - 1, 4 * x.size(-1) - 1], device=x.device, dtype=x.dtype)
    t2[:, :, 1: -1, 1:-1] = t1
    t2[:, :, 0, :] = 2 * t2[:, :, 1, :] - t2[:, :, 2, :]
    t2[:, :, -1, :] = 2 * t2[:, :, -2, :] - t2[:, :, -3, :]
    t2[:, :, :, 0] = 2 * t2[:, :, :, 1] - t2[:, :, :, 2]
    t2[:, :, :, -1] = 2 * t2[:, :, :, -2] - t2[:, :, :, -3]
    out = t2[:, :, ::2, ::2]
    return out


def _shift(x: torch.Tensor, shift: list) -> torch.Tensor:
    r""" Circular shift 2D matrix samples by OFFSET (a [Y,X] 2-vector), such that  RES(POS) = MTX(POS-OFFSET).

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        shift: Offset list.

    Returns:
        The circular shiftet tensor.
    """
    x_shifted = torch.cat((x[..., -shift[0]:, :], x[..., :-shift[0], :]), dim=-2)
    x_shifted = torch.cat((x_shifted[..., -shift[1]:], x_shifted[..., :-shift[1]]), dim=-1)
    return x_shifted
