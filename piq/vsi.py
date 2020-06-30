r"""Implemetation of Visual Saliency-induced Index
Code is based on MATLAB version for computations in pixel domain
https://sse.tongji.edu.cn/linzhang/IQA/VSI/VSI.htm

References:
    https://ieeexplore.ieee.org/document/6873260
"""

import torch
from torch.nn.modules.loss import _Loss
from torch.nn.functional import conv2d, avg_pool2d, interpolate
from .utils import _validate_input, _adjust_dimensions


def vsi(prediction: torch.Tensor, target: torch.Tensor, reduction: str = 'mean', data_range: float = 1.,
        c1: float = 1.27, c2: float = 386., c3: float = 130., alpha: float = 0.4, beta: float = 0.02,
        omega_0: float = 0.021, sigma_f: float = 1.34, sigma_d: float = 145., sigma_c: float = 0.001) -> torch.Tensor:
    r"""Compute Visual Saliency-induced Index for a batch of images.

        Both inputs supposed to have RGB order.
        Args:
            prediction: Batch of predicted images with shape (batch_size x channels x H x W)
            target: Batch of target images with shape  (batch_size x channels x H x W)
            reduction: Reduction over samples in batch: "mean"|"sum"|"none"
            data_range: Value range of input images (usually 1.0 or 255). Default: 1.0
            c1: coefficient to calculate saliency component of VSI
            c2: coefficient to calculate gradient component of VSI
            c3: coefficient to calculate color component of VSI
            alpha: power for gradient component of VSI
            beta: power for color component of VSI
            omega_0: coefficient to get log Gabor filter at SDSP
            sigma_f: coefficient to get log Gabor filter at SDSP
            sigma_d: coefficient to get SDSP
            sigma_c: coefficient to get SDSP

        Returns:
            VSI: Index of similarity between two images. Usually in [0, 1] interval.
        Note:
            See https://ieeexplore.ieee.org/document/6873260 for details.
        """

    _validate_input(input_tensors=(prediction, target), allow_5d=False)
    prediction, target = _adjust_dimensions(input_tensors=(prediction, target))
    assert prediction.size(-3) == 3 and target.size(-3) == 3, 'Expected RGB images, got images with ' \
                                                              '{prediction.size(-3)} and {target.size(-3)} channels'

    prediction = prediction * 255. / data_range
    target = target * 255. / data_range

    vs_prediction = sdsp(prediction, data_range=255., omega_0=omega_0,
                         sigma_f=sigma_f, sigma_d=sigma_d, sigma_c=sigma_c)
    vs_target = sdsp(target, data_range=255., omega_0=omega_0, sigma_f=sigma_f,
                     sigma_d=sigma_d, sigma_c=sigma_c)

    # Weitghts to translate from RGB colour space to LMN, see https://ieeexplore.ieee.org/document/6873260
    weights_RGB_to_LMN = torch.tensor([[0.06, 0.63, 0.27],
                                       [0.30, 0.04, -0.35],
                                       [0.34, -0.6, 0.17]]).to(target)

    prediction_LMN = torch.matmul(prediction.permute(0, 2, 3, 1), weights_RGB_to_LMN.t()).permute(0, 3, 1, 2)
    target_LMN = torch.matmul(target.permute(0, 2, 3, 1), weights_RGB_to_LMN.t()).permute(0, 3, 1, 2)

    kernel_size = max(1, round(min(vs_prediction.size()[-2:]) / 256))
    vs_prediction = avg_pool2d(vs_prediction.unsqueeze(1), kernel_size=kernel_size).squeeze(1)
    vs_target = avg_pool2d(vs_target.unsqueeze(1), kernel_size=kernel_size).squeeze(1)

    prediction_LMN = avg_pool2d(prediction_LMN, kernel_size=kernel_size)
    target_LMN = avg_pool2d(target_LMN, kernel_size=kernel_size)

    gm_prediction = scharr_grad_map(prediction_LMN[:, 1].unsqueeze(1)).squeeze(1)
    gm_target = scharr_grad_map(target_LMN[:, 1].unsqueeze(1)).squeeze(1)

    s_vs = (2 * vs_prediction * vs_target + c1) / (vs_prediction.pow(2) + vs_target.pow(2) + c1)
    s_gm = (2 * gm_prediction * gm_target + c2) / (gm_prediction.pow(2) + gm_target.pow(2) + c2)

    s_m = (2 * prediction_LMN[:, 1] * target_LMN[:, 1] + c3) / (
            prediction_LMN[:, 1].pow(2) + target_LMN[:, 1].pow(2) + c3)
    s_n = (2 * prediction_LMN[:, 2] * target_LMN[:, 2] + c3) / (
            prediction_LMN[:, 2].pow(2) + target_LMN[:, 2].pow(2) + c3)
    s_c = s_m * s_n

    s = s_vs * s_gm.pow(alpha) * s_c.relu().pow(beta)

    vs_max = torch.max(vs_prediction, vs_target)

    eps = torch.finfo(vs_max.dtype).eps
    output = ((s * vs_max).sum(dim=(-1, -2)) + eps) / (vs_max.sum(dim=(-1, -2)) + eps)

    if reduction == 'none':
        return output
    return {'mean': torch.mean,
            'sum': torch.sum
            }[reduction](output, dim=0)


class VSILoss(_Loss):
    def __init__(self, reduction: str = 'mean', c1: float = 1.27, c2: float = 386., c3: float = 130.,
                 alpha: float = 0.4, beta: float = 0.02, data_range: float = 1.,
                 omega_0: float = 0.021, sigma_f: float = 1.34, sigma_d: float = 145., sigma_c: float = 0.001) -> None:

        r"""Creates a criterion that measures Visual Saliency-induced Index error between
            each element in the input and target.

            The sum operation still operates over all the elements, and divides by :math:`n`.

            The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

            Args:
                reduction: Reduction over samples in batch: "mean"|"sum"|"none"
                data_range: Value range of input images (usually 1.0 or 255). Default: 1.0
                c1: coefficient to calculate saliency component of VSI
                c2: coefficient to calculate gradient component of VSI
                c3: coefficient to calculate color component of VSI
                alpha: power for gradient component of VSI
                beta: power for color component of VSI
                omega_0: coefficient to get log Gabor filter at SDSP
                sigma_f: coefficient to get log Gabor filter at SDSP
                sigma_d: coefficient to get SDSP
                sigma_c: coefficient to get SDSP

            Shape:
                - Input: Required to be 2D (H, W), 3D (C,H,W), 4D (N,C,H,W), channels first.
                - Target: Required to be 2D (H, W), 3D (C,H,W), 4D (N,C,H,W), channels first.

            Examples::

                >>> loss = VSILoss()
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
        super().__init__()
        self.reduction = reduction
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.alpha = alpha
        self.beta = beta
        self.data_range = data_range
        self.omega_0 = omega_0
        self.sigma_f = sigma_f
        self.sigma_d = sigma_d
        self.sigma_c = sigma_c
        pass

    def forward(self, prediction, target):
        r"""Computation of VSI as a loss function.

            Args:
                prediction: Tensor of prediction of the network.
                target: Reference tensor.

            Returns:
                Value of VSI loss to be minimized. 0 <= VSI loss <= 1.
        """

        return 1. - vsi(prediction=prediction, target=target, reduction=self.reduction, data_range=self.data_range,
                        c1=self.c1, c2=self.c2, c3=self.c3, alpha=self.alpha, beta=self.beta,
                        omega_0=self.omega_0, sigma_f=self.sigma_f, sigma_d=self.sigma_d, sigma_c=self.sigma_c)


def scharr_grad_map(x: torch.Tensor) -> torch.Tensor:
    # Schaee filters are used to get gradient maps
    filter_x = torch.tensor([[3., 0., -3.],
                             [10., 0., -10.],
                             [3., 0., -3.]]).to(x) / 16

    filter_y = torch.tensor([[3., 10., 3.],
                             [0., 0., 0.],
                             [-3., -10., -3.]]).to(x) / 16

    padding = filter_x.size(-1) // 2

    gm_x = conv2d(x, filter_x.view(1, 1, *filter_x.size()), padding=padding)
    gm_y = conv2d(x, filter_y.view(1, 1, *filter_y.size()), padding=padding)

    return torch.sqrt(gm_x.pow(2) + gm_y.pow(2))


def sdsp(x: torch.Tensor, data_range: float = 255., omega_0: float = 0.021, sigma_f: float = 1.34,
         sigma_d: float = 145., sigma_c: float = 0.001) -> torch.Tensor:
    size = x.size()

    size_to_use = (256, 256)
    x = interpolate(input=x, size=size_to_use, mode='bilinear', align_corners=True)
    x_lab = rgb2lab(x, data_range=data_range)

    x_fft = torch.fft(torch.stack([x_lab, torch.zeros_like(x_lab)], dim=-1), 2)
    lg = log_gabor(size_to_use, omega_0, sigma_f).to(x_fft).view(1, 1, *size_to_use, 1)
    x_ifft_real = torch.ifft(x_fft * lg, 2)[..., 0]
    s_f = x_ifft_real.pow(2).sum(dim=1).sqrt()

    coordinates = torch.stack((torch.meshgrid([torch.arange(0, size_to_use[0]) - (size[0] - 1) / 2,
                                               torch.arange(0, size_to_use[1]) - (size[1] - 1) / 2])), dim=0)
    s_d = torch.exp(-torch.sum(coordinates, dim=0) / sigma_d ** 2)

    eps = torch.finfo(x_lab.dtype).eps
    min_x = x_lab.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values
    max_x = x_lab.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values
    normilized = (x_lab - min_x) / (max_x - min_x + eps)

    norm = normilized[:, 1:].pow(2).sum(dim=1)
    s_c = 1 - torch.exp(-norm / sigma_c ** 2)

    vs_m = interpolate((s_f * s_d * s_c).unsqueeze(1), size[-2:], mode='bilinear', align_corners=True).squeeze(1)

    min_vs_m = vs_m.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values
    max_vs_m = vs_m.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values
    return (vs_m - min_vs_m) / (max_vs_m - min_vs_m + eps)


def rgb2xyz(x: torch.Tensor) -> torch.Tensor:
    """
    Translates image from RGB color space to XYZ
    Input image in [0, 1] range.
    """

    mask_below = x <= 0.04045
    mask_above = x > 0.04045

    tmp = x / 12.92 * mask_below + torch.pow((x + 0.055) / 1.055, 2.4) * mask_above

    weights_RGB_to_XYZ = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                                       [0.2126729, 0.7151522, 0.0721750],
                                       [0.0193339, 0.1191920, 0.9503041]]).to(x)

    x_xyz = torch.matmul(tmp.permute(0, 2, 3, 1), weights_RGB_to_XYZ.t()).permute(0, 3, 1, 2)

    return x_xyz


def xyz2lab(x: torch.Tensor) -> torch.Tensor:
    epsilon = 0.008856
    kappa = 903.3
    illuminants_d50 = torch.tensor([0.9642119944211994, 1., 0.8251882845188288]).to(x).view(1, 3, 1, 1)

    tmp = x / illuminants_d50

    mask_below = tmp <= epsilon
    mask_above = tmp > epsilon
    tmp = torch.pow(tmp, 1. / 3.) * mask_above + (kappa * tmp + 16.) / 116. * mask_below

    weights_XYZ_to_LAB = torch.tensor([[0, 116., 0],
                                       [500., -500., 0],
                                       [0, 200., -200.]]).to(x)
    bias_XYZ_to_LAB = torch.tensor([-16., 0., 0.]).to(x).view(1, 3, 1, 1)

    x_lab = torch.matmul(tmp.permute(0, 2, 3, 1), weights_XYZ_to_LAB.t()).permute(0, 3, 1, 2) + bias_XYZ_to_LAB
    return x_lab


def rgb2lab(x: torch.Tensor, data_range: float = 255.) -> torch.Tensor:
    return xyz2lab(rgb2xyz(x / data_range))


def log_gabor(size: tuple, omega_0: float = 0.021, sigma_f: float = 1.34) -> torch.Tensor:
    xx, yy = torch.meshgrid((torch.arange(0, size[0], dtype=torch.float32) - size[0] // 2) / (size[0] - size[0] % 2),
                            (torch.arange(0, size[1], dtype=torch.float32) - size[1] // 2) / (size[1] - size[1] % 2))

    mask = xx.pow(2) + yy.pow(2) <= 0.25
    xx = xx * mask
    yy = yy * mask

    xx = ifftshift(xx)
    yy = ifftshift(yy)

    r = (xx.pow(2) + yy.pow(2)).sqrt()
    r[0, 0] = 1

    lg = torch.exp((-(r / omega_0).log().pow(2)) / (2 * sigma_f ** 2))
    lg[0, 0] = 0
    return lg


def ifftshift(x: torch.Tensor):
    shift = [-(ax // 2) for ax in x.size()]
    return torch.roll(x, shift, tuple(range(len(shift))))
