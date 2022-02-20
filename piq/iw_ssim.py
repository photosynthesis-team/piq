r""" This module implements Information Content Weighted Structural Similarity Index (IW-SSIM) index in PyTorch.

It is based on original MATLAB code from authors [1] and PyTorch port by Jack Guo Xy [2].
References:
 [1] https://ece.uwaterloo.ca/~z70wang/research/iwssim/iwssim_iwpsnr.zip
 [2] https://github.com/Jack-guo-xy/Python-IW-SSIM
"""


import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from piq.utils import _validate_input, _reduce
from piq.functional import gaussian_filter, binomial_filter1d, average_filter2d, rgb2yiq
from typing import Union, Optional, Tuple
import math


def information_weighted_ssim(x: torch.Tensor, y: torch.Tensor, data_range: Union[int, float] = 1.,
                              kernel_size: int = 11, kernel_sigma: float = 1.5, k1: float = 0.01, k2: float = 0.03,
                              parent: bool = True, blk_size: int = 3, sigma_nsq: float = 0.4,
                              scale_weights: Optional[torch.Tensor] = None,
                              reduction: str = 'mean') -> torch.Tensor:
    r"""

    Args:
        x:
        y:
        data_range:
        kernel_size:
        kernel_sigma:
        k1:
        k2:
        parent:
        scale_weights:
        reduction:

    Returns:

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

    levels = scale_weights.size(0)

    min_size = (kernel_size - 1) * 2 ** (levels - 1) + 1
    if x.size(-1) < min_size or x.size(-2) < min_size:
        raise ValueError(f'Invalid size of the input images, expected at least {min_size}x{min_size}.')

    bound = math.ceil((kernel_size - 1) / 2)  # Ceil
    bound1 = bound - math.floor((blk_size - 1) / 2)  # floor
    gauss_kernel = gaussian_filter(kernel_size, kernel_sigma).repeat(x.size(1), 1, 1, 1).to(x)

    pyramid_kernel_size = 5
    bin_filter = binomial_filter1d(kernel_size=pyramid_kernel_size).to(x) * 2**0.5
    #print(bin_filter.size())
    # Blur and downsample
    #lo_x = _blur_and_downsample(x, bin_filter)
    #lo_y = _blur_and_downsample(y, bin_filter)

    # Upsample and blur
    #x_diff_old = x - _upsample_and_blur(lo_x, bin_filter)[:, :, :x.size(-2), :x.size(-1)]
    #y_diff_old = y - _upsample_and_blur(lo_y, bin_filter)[:, :, :y.size(-2), :y.size(-1)]

    lo_x, x_diff_old = _pyr_step(x, bin_filter)
    lo_y, y_diff_old = _pyr_step(y, bin_filter)

    #print(lo_x.size())
    #print(lo_x[0, 0, :10, :10])
    #print(x_diff_old.size())
    #print(x_diff_old[0,0,:10,:10])
    #print(lo_y.size())
    #print(lo_y[0, 0, :10, :10])
    #print(y_diff_old.size())
    #print(y_diff_old[0, 0, :10, :10])

    x = lo_x
    y = lo_y
    wmcs = []

    for i in range(levels):
        print(i)
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

            iw_map = iw_map[:, :, bound1:-bound1, bound1:-bound1]

        elif i == levels - 1:
            iw_map = torch.ones_like(cs_map)
            cs_map = ssim_map

        else:
            iw_map = _information_content(x=x_diff_old, y=y_diff_old, y_parent=None, kernel_size=blk_size,
                                          sigma_nsq=sigma_nsq)
            iw_map = iw_map[:, :, bound1:-bound1, bound1:-bound1]

        wmcs.append(torch.sum(cs_map * iw_map, dim=(-2, -1)) / torch.sum(iw_map, dim=(-2, -1)))

        x_diff_old = x_diff
        y_diff_old = y_diff

    # TODO: It contains negative values leading to NaN result.
    wmcs = torch.stack(wmcs, dim=0)

    score = torch.prod((wmcs ** scale_weights.view(-1, 1, 1)), dim=0)[:, 0]

    return _reduce(x=score, reduction=reduction)


class InformationWeightedSSIMLoss(_Loss):
    r"""

    """

    def __init__(self, data_range: Union[int, float] = 1., reduction: str = 'mean'):
        self.reduction = reduction
        self.data_range = data_range

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return information_weighted_ssim(x=x, y=y, data_range=self.data_range, reduction=self.reduction)


def _pyr_step(x: torch.Tensor, kernel:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""

    Args:
        x:
        kernel:

    Returns:

    """
    # Blur and Downsampling
    up_pad = (kernel.size(-1) - 1) // 2  # 5 -> 2, 4 -> 1
    down_pad = kernel.size(-1) - 1 - up_pad  # 5 -> 2, 4 -> 2
    kernel_t = kernel.transpose(-2, -1)
    lo_x = x
    if x.size(-1) > 1:
        lo_x = F.pad(lo_x, pad=[up_pad, down_pad, 0, 0], mode='reflect')
        lo_x = F.conv2d(input=lo_x, weight=kernel.unsqueeze(0), padding='valid')[:, :, :, ::2]
    if x.size(-2) > 1:
        lo_x = F.pad(lo_x, pad=[0, 0, up_pad, down_pad], mode='reflect')
        lo_x = F.conv2d(input=lo_x, weight=kernel_t.unsqueeze(0), padding='valid')[:, :, ::2, :]

    # Upsampling and Blur
    up_pad = (kernel.size(-1) - 1) // 2  # 5 -> 2, 4 -> 1
    down_pad = kernel.size(-1) - 1 - up_pad  # 5 -> 2, 4 -> 2
    hi_x = lo_x

    if x.size(-1) > 1:
        upsampling_kernel = torch.tensor([[[[1., 0.]]]], dtype=x.dtype, device=x.device)
        hi_x = F.conv_transpose2d(input=hi_x, weight=upsampling_kernel, stride=(1, 2), padding=0)
        hi_x = F.pad(hi_x, pad=[up_pad, down_pad, 0, 0], mode='reflect')
        hi_x = F.conv2d(input=hi_x, weight=kernel.unsqueeze(0), padding='valid')[:, :, :, :x.size(-1)]
    if x.size(-2) > 1:
        upsampling_kernel = torch.tensor([[[[1.], [0.]]]], dtype=x.dtype, device=x.device)
        hi_x = F.conv_transpose2d(input=hi_x, weight=upsampling_kernel, stride=(2, 1), padding=0)
        hi_x = F.pad(hi_x, pad=[0, 0, up_pad, down_pad], mode='reflect')
        hi_x = F.conv2d(input=hi_x, weight=kernel_t.unsqueeze(0), padding='valid')[:, :, :x.size(-2), :]

    hi_x = x - hi_x
    return lo_x, hi_x


def _ssim_per_channel(x: torch.Tensor, y: torch.Tensor, kernel: torch.Tensor,
                      data_range: Union[float, int] = 1., k1: float = 0.01,
                      k2: float = 0.03) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Calculate Structural Similarity (SSIM) index for X and Y per channel.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        kernel: 2D Gaussian kernel.
        data_range: Maximum value range of images (usually 1.0 or 255).
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:

    """
    if x.size(-1) < kernel.size(-1) or x.size(-2) < kernel.size(-2):
        raise ValueError(f'Kernel size can\'t be greater than actual input size. Input size: {x.size()}. '
                         f'Kernel size: {kernel.size()}')

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
                         kernel_size: int = 3, sigma_nsq: float = 0.4):
    r""""""

    EPS = torch.finfo(x.dtype).eps
    n_channels = x.size(1)
    kernel = average_filter2d(kernel_size=kernel_size).repeat(x.size(1), 1, 1, 1).to(x)
    mu_x = F.conv2d(input=x, weight=kernel, padding='same', groups=n_channels)
    mu_y = F.conv2d(input=y, weight=kernel, padding='same', groups=n_channels)

    mu_xx = mu_x ** 2
    mu_yy = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_xx = F.conv2d(x ** 2, weight=kernel, stride=1, padding='same', groups=n_channels) - mu_xx
    sigma_yy = F.conv2d(y ** 2, weight=kernel, stride=1, padding='same', groups=n_channels) - mu_yy
    sigma_xy = F.conv2d(x * y, weight=kernel, stride=1, padding='same', groups=n_channels) - mu_xy

    sigma_xx = F.relu(sigma_xx)
    sigma_yy = F.relu(sigma_yy)

    g = sigma_xy / (sigma_yy + EPS)
    vv = sigma_xx - g * sigma_xy
    g = g.masked_fill(sigma_yy < EPS, 0)
    vv[sigma_yy < EPS] = sigma_xx[sigma_yy < EPS]
    sigma_yy = sigma_yy.masked_fill(sigma_yy < EPS, 0)
    g = g.masked_fill(sigma_xx < EPS, 0)
    vv = vv.masked_fill(sigma_xx < EPS, 0)

    block = [kernel_size, kernel_size]

    nblv = y.size(-2) - block[0] + 1
    nblh = y.size(-1) - block[1] + 1
    nexp = nblv * nblh
    N = block[0] * block[1]


    assert block[0] % 2 == 1 and block[1] % 2  == 1, f'Expected odd block dimensions, got {block}'

    Ly = (block[0] - 1) // 2
    Lx = (block[1] - 1) // 2

    if y_parent is not None:
        # upscale y_parent and cut to the size of y

        y_parent_up = _image_enlarge(y_parent)[:, :, :y.size(-2), :y.size(-1)]
        N = N + 1

    Y = torch.zeros(y.size(0), y.size(1), nexp, N)

    n = -1
    for ny in range(-Ly, Ly+1):
        for nx in range(-Lx, Lx + 1):
            n = n + 1
            foo = _shift(y, [ny, nx])
            foo = foo[:, :, Ly:Ly + nblv, Lx:Lx + nblh]
            Y[..., n] = foo.flatten(start_dim=-2, end_dim=-1)

    if y_parent is not None:
        n = n+1
        foo = y_parent_up
        foo = foo[:, :, Ly:Ly + nblv, Lx:Lx + nblh]
        Y[..., n] = foo.flatten(start_dim=-2, end_dim=-1)

    C_u = torch.matmul(Y.transpose(-2, -1), Y) / nexp
    eig_values, eig_vectors = torch.linalg.eigh(C_u)

    sum_eig_values = torch.sum(eig_values, dim=-1).view(y.size(0), y.size(1), 1, 1)
    non_zero_eig_values_matrix = torch.diag_embed(eig_values * (eig_values > 0))
    sum_non_zero_eig_values = torch.sum(non_zero_eig_values_matrix, dim=(-2, -1), keepdim=True)

    L = non_zero_eig_values_matrix * sum_eig_values / (sum_non_zero_eig_values + (sum_non_zero_eig_values == 0))

    C_u = torch.matmul(torch.matmul(eig_vectors, L), eig_vectors.transpose(-2, -1))

    C_u_inv = torch.linalg.inv(C_u)

    ss = torch.matmul(Y, C_u_inv) * Y / N
    ss = torch.sum(ss, dim=-1, keepdim=True)
    ss = ss.view(y.size(0), y.size(1), nblv, nblh)
    g = g[:, :, Ly: Ly + nblv, Lx: Lx + nblh]
    vv = vv[:, :, Ly: Ly + nblv, Lx: Lx + nblh]

    # Calculate mutual information
    scaled_eig_values = torch.diagonal(L, offset=0, dim1=-2, dim2=-1).unsqueeze(2).unsqueeze(3)

    iw_map = torch.sum(torch.log2(1 + ((vv.unsqueeze(-1) + (1 + g.unsqueeze(-1) * g.unsqueeze(-1)) * sigma_nsq)
        * ss.unsqueeze(-1) * scaled_eig_values + sigma_nsq * vv.unsqueeze(-1)) / (sigma_nsq * sigma_nsq)), dim=-1)

    iw_map[iw_map < EPS] = 0

    return iw_map


def _image_enlarge(x: torch.Tensor) -> torch.Tensor:
    t1 = F.interpolate(x, size=(int(4 * x.size(-2) - 3), int(4 * x.size(-1) - 3)), mode='bilinear', align_corners=False)
    t2 = torch.zeros([x.size(0), 1, 4 * x.size(-2) - 1, 4 * x.size(-1) - 1]).to(x)
    t2[:, :, 1: -1, 1:-1] = t1
    t2[:, :, 0, :] = 2 * t2[:, :, 1, :] - t2[:, :, 2, :]
    t2[:, :, -1, :] = 2 * t2[:, :, -2, :] - t2[:, :, -3, :]
    t2[:, :, :, 0] = 2 * t2[:, :, :, 1] - t2[:, :, :, 2]
    t2[:, :, :, -1] = 2 * t2[:, :, :, -2] - t2[:, :, :, -3]
    out = t2[:, :, ::2, ::2]
    return out


def _shift(x: torch.Tensor, shift: list) -> torch.Tensor:
    tmp = torch.cat((x[..., -shift[0]:, :], x[..., :-shift[0], :]), dim=-2)
    tmp = torch.cat((tmp[..., -shift[1]:], tmp[..., :-shift[1]]), dim=-1)
    return tmp