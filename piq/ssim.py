r""" This module implements Structural Similarity (SSIM) index in PyTorch.

    Description:
        Suppose $x$ and $y$ are two non-negative image signals, which have been aligned with each other (e.g., spatial patches extracted from each image). If considered one of the signals to have perfect quality, then the similarity measure can serve as a quantitative measure-ment of the quality of the second signal. The system separates the task of similarity measurement into three comparisons:  luminance, contrast and structure.  First, the luminance of each signal is compared. Assuming discrete signals, this is estimated as the mean intensity:
        $$
        \mu_y=\frac{1}{N}\sum_{i = 1}^{n}x_i. \qquad (1)
        $$
        The luminance comparison function $l(x,y)$ is then a function of $\mu_x$ and $\mu_y$, and remove the mean intensity from the signal. In discrete form, the resulting signal $x−\mu_x$ corresponds to the projection of vector $x$ on to the hyperplane defined by
        $$
        \sum_{i = 1}^{n}x_i=0. \qquad (2)
        $$
        The standard deviation (the square root of variance) as an estimate of the signal contrast. An unbiased estimatein discrete form is given by
        $$
        \sigma_{x}=\left(\frac{1}{N-1} \sum_{i=1}^{N}\left(x_{i}-\mu_{x}\right)^{2}\right)^{1 / 2} \qquad (3)
        $$
        The contrast comparison $c(x,y)$ is then the comparison of $\sigma_{x}$ and $\sigma_y$. Third, the signal is normalized (divided) by its own standard deviation, so that the two signals being compared have unit standard deviation. The structure comparison $s(x,y)$ is conducted on these normalized signals $(x−\mu_{x})/\sigma_{x}$ and $(y−\mu_{y})/\sigma_{y}$.
        Finally, the three components are combined to yield anoverall similarity measure:
        $$
        S(x,y) =f(l(x,y), c(x,y), s(x,y)). \qquad (4)
        $$
        An important point is that the three components are relatively independent. For example, the change of luminance and/or contrast will not affect the structures of images. In order to complete the definition of the similarity measure in Eq.  (4), needs to define the three functions $l(x,y),c(x,y),s(x,y)$, as well as the combination function $f(·)$.  We also would like the similarity measure to satisfy the following conditions:
        1. Symmetry: $S(x,y) =S(y,x)$;
        2. Boundedness: $S(x,y)\leq 1$;
        3. Unique maximum: $S(x,y) = 1$ if and only if $x=y$(indiscrete representations, $x_{i}=y_i$ for all $i= 1,2,···, N$).

        For luminance comparison defined
        $$
        l(x,y) =\frac{2\mu_x\mu_y+C_1}{{\mu_x}^2+{\mu_y}^2+C_1} \qquad (5)
        $$
        where the constant $C_1$ is included to avoid instability when {\mu_x}^2+{\mu_y}^2 is very close to zero. Specifically,  choose $$C_1= {(K_1L)}^2, \qquad (6)$$
        where $L$ is the dynamic range of the pixel values (255 for 8-bit grayscale images), and $K_1 \ll 1$ is a small constant. Similar considerations also apply to contrast comparisonand structure comparison described later. Eq. (5) is easily seen to obey the three properties listed above. Equation (5) is also qualitatively consistent with We-ber’s law, which has been widely used to model light adaptation (also called luminance masking) in the HVS. According to Weber’s law, the magnitude of a just-noticeable luminance change $\Delta I$ is approximately proportional to the background luminance $I$ for a wide range of luminance values. In other words, the HVS is sensitive to therelative luminance change, and not the absolute luminance change. Letting $R$ represent the size of luminance change relativ eto background luminance, we rewrite the luminance of the distorted signal as $\mu_y= (1 +R)\mu_x$. Substituting this into Eq. (5) gives 
        $$
        l(x,y) =\frac{2(1 +R)}{1 + {(1 +R)}^2+C_1/{\mu_x}^2} \qquad (7)
        $$
        If we assume $C_1$ is small enough (relative to ${\mu_x}^2$) to beignored, then $l(x,y)$ is a function only of $R$, qualitatively consistent with Weber’s law. The contrast comparison function takes a similar form:
        $$
        c(x,y) =\frac {2\sigma_x\sigma_y+C_2} {{\sigma_x}^2+{\sigma_y}^2+C_2}, \qquad (8)
        $$
        where $C_2= (K_2L)^2$, and $K_2 \ll 1$. This definition againsatisfies the three properties listed above.  An importantfeature of this function is that with the same amount of contrast change $\Delta\sigma=\sigma_y−\sigma_x$, this measure is less sensitive to the case of high base contrast $\sigma_x$ than low base contrast. This is consistent with the contrast masking feature of the HVS.

        Structure comparison is conducted after luminance subtraction and variance normalization.  Specifically, associate the two unit vectors $(x−\mu_x)/\sigma_x$ and $(y−\mu_y)/\sigma_y$, each lying in the hyperplane defined by Eq. (2), with thestructure of the two images. The correlation (inner product) between these is a simple and effective measure toquantify the structural similarity. Notice that the corre-lation between $(x−\mu_x)/\sigma_x$ and $(y−\mu_y)/\sigma_y$ is equivalent to the correlation coefficient between $x$ and $y$. Thus, wedefine the structure comparison function as follows:
        $$
        s(x,y) =\frac {\sigma_{xy}+C_3}{\sigma_x\sigma_y+C_3}. \qquad (9)
        $$
        As in the luminance and contrast measures, we have introduced a small constant in both denominator and numerator.  In discrete form, $\sigma_{xy}$ can be estimated as:
        $$
        \sigma_{xy}=\frac{1}{N−1N}\sum_{i = 1}^{n}(x_i−\mu_x)(y_i−\nu_y). \qquad (10)
        $$
        Geometrically, the correlation coefficient corresponds to the cosine of the angle between the vectors $x−\mu_x$ and $y−\nu_y$. Note also that $s(x,y)$ can take on negative values. Finally, combine the three comparisons of Eqs. (5),(8) and (9) and name the resulting similarity measure the **Structural SIMilarity (SSIM)** index between signals $x$ and $y$:
        $$
        SSIM(x,y) = {[l(x,y)]}^{\alpha}·{[c(x,y)]}^{\beta}·{[s(x,y)]}^{\gamma}, \qquad (11)
        $$
        where $\alpha >0$, $\beta >0$ and $\gamma >0$ are parameters used to adjust the relative importance of the three components. It is easy to verify that this definition satisfies the three conditions given above. In order to simplify the expression,  set $\alpha = \beta = \gamma = 1$ and $C_3=C_2/2$. This results in a specific form of the SSIM index:
        $$
        \operatorname{SSIM}(\mathrm{x}, \mathrm{y})=\frac{\left(2 \mu_{x} \mu_{y}+C_{1}\right)\left(2 \sigma_{x y}+C_{2}\right)}{\left(\mu_{x}^{2}+\mu_{y}^{2}+C_{1}\right)\left(\sigma_{x}^{2}+\sigma_{y}^{2}+C_{2}\right)} \qquad (12)
        $$


Implementation of classes and functions from this module are inspired by Gongfan Fang's (@VainF) implementation:
https://github.com/VainF/pytorch-msssim

and implementation of one of pull requests to the PyTorch by Kangfu Mei (@MKFMIKU):
https://github.com/pytorch/pytorch/pull/22289/files
"""
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from piq.utils import _adjust_dimensions, _validate_input
from piq.functional import gaussian_filter


def ssim(x: torch.Tensor, y: torch.Tensor, kernel_size: int = 11, kernel_sigma: float = 1.5,
         data_range: Union[int, float] = 1., reduction: str = 'mean', full: bool = False,
         k1: float = 0.01, k2: float = 0.03) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Interface of Structural Similarity (SSIM) index.
    Inputs supposed to be in range [0, data_range] with RGB channels order for colour images.

    Args:
        x: Tensor with shape 2D (H, W), 3D (C, H, W), 4D (N, C, H, W) or 5D (N, C, H, W, 2).
        y: Tensor with shape 2D (H, W), 3D (C, H, W), 4D (N, C, H, W) or 5D (N, C, H, W, 2).
        kernel_size: The side-length of the sliding window used in comparison. Must be an odd value.
        kernel_sigma: Sigma of normal distribution.
        data_range: Value range of input images (usually 1.0 or 255).
        reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        full: Return cs map or not.
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
           DOI: `10.1109/TIP.2003.819861`
    """
    _validate_input(
        input_tensors=(x, y), allow_5d=True, kernel_size=kernel_size, scale_weights=None, data_range=data_range)
    x, y = _adjust_dimensions(input_tensors=(x, y))

    x = x.type(torch.float32)
    y = y.type(torch.float32)
        
    x = x / data_range
    y = y / data_range

    kernel = gaussian_filter(kernel_size, kernel_sigma).repeat(x.size(1), 1, 1, 1).to(y)
    _compute_ssim_per_channel = _ssim_per_channel_complex if x.dim() == 5 else _ssim_per_channel
    ssim_map, cs_map = _compute_ssim_per_channel(x=x, y=y, kernel=kernel, data_range=data_range, k1=k1, k2=k2)
    ssim_val = ssim_map.mean(1)
    cs = cs_map.mean(1)

    if reduction != 'none':
        reduction_operation = {'mean': torch.mean,
                               'sum': torch.sum}
        ssim_val = reduction_operation[reduction](ssim_val, dim=0)
        cs = reduction_operation[reduction](cs, dim=0)

    if full:
        return ssim_val, cs

    return ssim_val


class SSIMLoss(_Loss):
    r"""Creates a criterion that measures the structural similarity index error between
    each element in the input :math:`x` and target :math:`y`.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        SSIM = \{ssim_1,\dots,ssim_{N \times C}\}\\
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
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        data_range: The difference between the maximum and minimum of the pixel value,
            i.e., if for image x it holds min(x) = 0 and max(x) = 1, then data_range = 1.
            The pixel value interval of both input and output should remain the same.

    Shape:
        - Input: 2D (H, W), 3D (C, H, W), 4D (N, C, H, W) or 5D (N, C, H, W, 2).
        - Target: 2D (H, W), 3D (C, H, W), 4D (N, C, H, W) or 5D (N, C, H, W, 2).

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
           DOI:`10.1109/TIP.2003.819861`
    """
    __constants__ = ['kernel_size', 'k1', 'k2', 'sigma', 'kernel', 'reduction']

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

    def forward(self,
                prediction: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        r"""Computation of Structural Similarity (SSIM) index as a loss function.

        Args:
            prediction: Tensor with shape 2D (H, W), 3D (C, H, W), 4D (N, C, H, W) or 5D (N, C, H, W, 2).
            target: Tensor with shape 2D (H, W), 3D (C, H, W), 4D (N, C, H, W) or 5D (N, C, H, W, 2).

        Returns:
            Value of SSIM loss to be minimized, i.e 1 - `ssim`. 0 <= SSIM loss <= 1. In case of 5D input tensors,
            complex value is returned as a tensor of size 2.
        """

        score = ssim(x=prediction, y=target, kernel_size=self.kernel_size, kernel_sigma=self.kernel_sigma,
                     data_range=self.data_range, reduction=self.reduction, full=False, k1=self.k1, k2=self.k2)
        return torch.ones_like(score) - score


def multi_scale_ssim(x: torch.Tensor, y: torch.Tensor, kernel_size: int = 11, kernel_sigma: float = 1.5,
                     data_range: Union[int, float] = 1., reduction: str = 'mean',
                     scale_weights: Optional[Union[Tuple[float], List[float], torch.Tensor]] = None,
                     k1: float = 0.01, k2: float = 0.03) -> torch.Tensor:
    r""" Interface of Multi-scale Structural Similarity (MS-SSIM) index.
    Inputs supposed to be in range [0, data_range] with RGB channels order for colour images.
    The size of the image should be at least (kernel_size - 1) * 2 ** (levels - 1) + 1.

    Args:
        x: Tensor with shape 2D (H, W), 3D (C, H, W), 4D (N, C, H, W) or 5D (N, C, H, W, 2).
        y: Tensor with shape 2D (H, W), 3D (C, H, W), 4D (N, C, H, W) or 5D (N, C, H, W, 2).
        kernel_size: The side-length of the sliding window used in comparison. Must be an odd value.
        kernel_sigma: Sigma of normal distribution.
        data_range: Value range of input images (usually 1.0 or 255).
        reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
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
           DOI:`10.1109/ACSSC.2003.1292216`
        .. [2] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
           (2004). Image quality assessment: From error visibility to
           structural similarity. IEEE Transactions on Image Processing,
           13, 600-612.
           https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
           DOI: `10.1109/TIP.2003.819861`
    """
    _validate_input(
        input_tensors=(x, y), allow_5d=True, kernel_size=kernel_size,
        scale_weights=scale_weights, data_range=data_range
    )
    x, y = _adjust_dimensions(input_tensors=(x, y))

    x = x.type(torch.float32)
    y = y.type(torch.float32)

    x = x / data_range
    y = y / data_range

    if scale_weights is None:
        scale_weights_from_ms_ssim_paper = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        scale_weights = scale_weights_from_ms_ssim_paper

    scale_weights_tensor = scale_weights if isinstance(scale_weights, torch.Tensor) else torch.tensor(scale_weights)
    scale_weights_tensor = scale_weights_tensor.to(y)
    kernel = gaussian_filter(kernel_size, kernel_sigma).repeat(x.size(1), 1, 1, 1).to(y)
    
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

    if reduction == 'none':
        return msssim_val

    return {'mean': torch.mean,
            'sum': torch.sum}[reduction](msssim_val, dim=0)


class MultiScaleSSIMLoss(_Loss):
    r"""Creates a criterion that measures the multi-scale structural similarity index error between
    each element in the input :math:`x` and target :math:`y`.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        MSSIM = \{mssim_1,\dots,mssim_{N \times C}\}, \\
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

    The size of the image should be (kernel_size - 1) * 2 ** (levels - 1) + 1.
    For colour images channel order is RGB.
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
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        data_range: The difference between the maximum and minimum of the pixel value,
            i.e., if for image x it holds min(x) = 0 and max(x) = 1, then data_range = 1.
            The pixel value interval of both input and output should remain the same.

    Shape:
        - Input: 2D (H, W), 3D (C, H, W), 4D (N, C, H, W) or 5D (N, C, H, W, 2).
        - Target: 2D (H, W), 3D (C, H, W), 4D (N, C, H, W) or 5D (N, C, H, W, 2).

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
           DOI:`10.1109/ACSSC.2003.1292216`
        .. [2] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
           (2004). Image quality assessment: From error visibility to
           structural similarity. IEEE Transactions on Image Processing,
           13, 600-612.
           https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
           DOI:`10.1109/TIP.2003.819861`
    """
    __constants__ = ['kernel_size', 'k1', 'k2', 'sigma', 'kernel', 'reduction']

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
        self.scale_weights = scale_weights if isinstance(scale_weights, torch.Tensor) else torch.tensor(scale_weights)
        self.kernel_size = kernel_size
        self.kernel_sigma = kernel_sigma
        self.k1 = k1
        self.k2 = k2
        self.data_range = data_range

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Computation of Multi-scale Structural Similarity (MS-SSIM) index as a loss function.
        The size of the image should be at least (kernel_size - 1) * 2 ** (levels - 1) + 1.
        For colour images channel order is RGB.

        Args:
            prediction: Tensor with shape 2D (H, W), 3D (C, H, W), 4D (N, C, H, W) or 5D (N, C, H, W, 2).
            target: Tensor with shape 2D (H, W), 3D (C, H, W), 4D (N, C, H, W) or 5D (N, C, H, W, 2).

        Returns:
            Value of MS-SSIM loss to be minimized, i.e. 1-`ms_sim`. 0 <= MS-SSIM loss <= 1. In case of 5D tensor,
            complex value is returned as a tensor of size 2.
        """

        score = multi_scale_ssim(x=prediction, y=target, kernel_size=self.kernel_size, kernel_sigma=self.kernel_sigma,
                                 data_range=self.data_range, reduction=self.reduction, scale_weights=self.scale_weights,
                                 k1=self.k1, k2=self.k2)
        return torch.ones_like(score) - score


def _ssim_per_channel(x: torch.Tensor, y: torch.Tensor, kernel: torch.Tensor,
                      data_range: Union[float, int] = 1., k1: float = 0.01,
                      k2: float = 0.03) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Calculate Structural Similarity (SSIM) index for X and Y per channel.

    Args:
        x: Tensor with shape (N, C, H, W).
        y: Tensor with shape (N, C, H, W).
        kernel: 2D Gaussian kernel.
        data_range: Value range of input images (usually 1.0 or 255).
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Full Value of Structural Similarity (SSIM) index.
    """

    if x.size(-1) < kernel.size(-1) or x.size(-2) < kernel.size(-2):
        raise ValueError(f'Kernel size can\'t be greater than actual input size. Input size: {x.size()}. '
                         f'Kernel size: {kernel.size()}')

    c1 = k1 ** 2
    c2 = k2 ** 2
    n_channels = x.size(1)
    mu1 = F.conv2d(x, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu2 = F.conv2d(y, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    compensation = 1.0
    sigma1_sq = compensation * (F.conv2d(x * x, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_sq)
    sigma2_sq = compensation * (F.conv2d(y * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu2_sq)
    sigma12 = compensation * (F.conv2d(x * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_mu2)

    # Set alpha = beta = gamma = 1.
    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map

    ssim_val = ssim_map.mean(dim=(-1, -2))
    cs = cs_map.mean(dim=(-1, -2))
    return ssim_val, cs


def _multi_scale_ssim(x: torch.Tensor, y: torch.Tensor, data_range: Union[int, float], kernel: torch.Tensor,
                      scale_weights_tensor: torch.Tensor, k1: float, k2: float) -> torch.Tensor:
    r"""Calculates Multi scale Structural Similarity (MS-SSIM) index for X and Y.

    Args:
        x: Tensor with shape (N, C, H, W).
        y: Tensor with shape (N, C, H, W).
        data_range: Value range of input images (usually 1.0 or 255).
        kernel: 2D Gaussian kernel.
        scale_weights_tensor: Weights for scaled SSIM
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Value of Multi scale Structural Similarity (MS-SSIM) index.
    """
    levels = scale_weights_tensor.size(0)
    min_size = (kernel.size(-1) - 1) * 2 ** (levels - 1) + 1
    if x.size(-1) < min_size or x.size(-2) < min_size:
        raise ValueError(f'Invalid size of the input images, expected at least {min_size}x{min_size}.')

    mcs = []
    ssim_val = None
    for iteration in range(levels):
        if iteration > 0:
            padding = max(x.shape[2] % 2, x.shape[3] % 2)
            x = F.pad(x, pad=[padding, 0, padding, 0], mode='replicate')
            y = F.pad(y, pad=[padding, 0, padding, 0], mode='replicate')
            x = F.avg_pool2d(x, kernel_size=2, padding=0)
            y = F.avg_pool2d(y, kernel_size=2, padding=0)

        ssim_val, cs = _ssim_per_channel(x, y, kernel=kernel, data_range=data_range, k1=k1, k2=k2)
        mcs.append(cs)

    # mcs, (level, batch)
    mcs_ssim = torch.relu(torch.stack(mcs[:-1] + [ssim_val], dim=0))

    # weights, (level)
    msssim_val = torch.prod((mcs_ssim ** scale_weights_tensor.view(-1, 1, 1)), dim=0).mean(1)

    return msssim_val


def _ssim_per_channel_complex(x: torch.Tensor, y: torch.Tensor, kernel: torch.Tensor,
                              data_range: Union[float, int] = 1., k1: float = 0.01,
                              k2: float = 0.03) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Calculate Structural Similarity (SSIM) index for Complex X and Y per channel.

    Args:
        x: Complex tensor with shape (N, C, H, W, 2).
        y: Complex tensor with shape (N, C, H, W, 2).
        kernel: 2-D gauss kernel.
        data_range: Value range of input images (usually 1.0 or 255).
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Full Value of Complex Structural Similarity (SSIM) index.
    """
    n_channels = x.size(1)
    if x.size(-2) < kernel.size(-1) or x.size(-3) < kernel.size(-2):
        raise ValueError(f'Kernel size can\'t be greater than actual input size. Input size: {x.size()}. '
                         f'Kernel size: {kernel.size()}')

    c1 = k1 ** 2
    c2 = k2 ** 2

    x_real = x[..., 0]
    x_imag = x[..., 1]
    y_real = y[..., 0]
    y_imag = y[..., 1]

    mu1_real = F.conv2d(x_real, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu1_imag = F.conv2d(x_imag, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu2_real = F.conv2d(y_real, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu2_imag = F.conv2d(y_imag, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu1_sq = mu1_real.pow(2) + mu1_imag.pow(2)
    mu2_sq = mu2_real.pow(2) + mu2_imag.pow(2)
    mu1_mu2_real = mu1_real * mu2_real - mu1_imag * mu2_imag
    mu1_mu2_imag = mu1_real * mu2_imag + mu1_imag * mu2_real

    compensation = 1.0

    x_sq = x_real.pow(2) + x_imag.pow(2)
    y_sq = y_real.pow(2) + y_imag.pow(2)
    x_y_real = x_real * y_real - x_imag * y_imag
    x_y_imag = x_real * y_imag + x_imag * y_real

    sigma1_sq = F.conv2d(x_sq, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_sq
    sigma2_sq = F.conv2d(y_sq, weight=kernel, stride=1, padding=0, groups=n_channels) - mu2_sq
    sigma12_real = F.conv2d(x_y_real, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_mu2_real
    sigma12_imag = F.conv2d(x_y_imag, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_mu2_imag
    sigma12 = torch.stack((sigma12_imag, sigma12_real), dim=-1)
    mu1_mu2 = torch.stack((mu1_mu2_real, mu1_mu2_imag), dim=-1)
    # Set alpha = beta = gamma = 1.
    cs_map = (sigma12 * 2 + c2 * compensation) / (sigma1_sq.unsqueeze(-1) + sigma2_sq.unsqueeze(-1) + c2 * compensation)
    ssim_map = (mu1_mu2 * 2 + c1 * compensation) / (mu1_sq.unsqueeze(-1) + mu2_sq.unsqueeze(-1) + c1 * compensation)
    ssim_map = ssim_map * cs_map

    ssim_val = ssim_map.mean(dim=(-2, -3))
    cs = cs_map.mean(dim=(-2, -3))

    return ssim_val, cs


def _multi_scale_ssim_complex(x: torch.Tensor, y: torch.Tensor, data_range: Union[int, float],
                              kernel: torch.Tensor, scale_weights_tensor: torch.Tensor, k1: float,
                              k2: float) -> torch.Tensor:
    r"""Calculate Multi scale Structural Similarity (MS-SSIM) index for Complex X and Y.

    Args:
        x: Complex tensor with shape (N, C, H, W, 2).
        y: Complex tensor with shape (N, C, H, W, 2).
        data_range: Value range of input images (usually 1.0 or 255).
        kernel: 2-D gauss kernel.
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Value of Complex Multi scale Structural Similarity (MS-SSIM) index.
    """
    levels = scale_weights_tensor.size(0)
    min_size = (kernel.size(-1) - 1) * 2 ** (levels - 1) + 1
    if x.size(-2) < min_size or x.size(-3) < min_size:
        raise ValueError(f'Invalid size of the input images, expected at least {min_size}x{min_size}.')
    mcs = []
    ssim_val = None
    for iteration in range(levels):
        x_real = x[..., 0]
        x_imag = x[..., 1]
        y_real = y[..., 0]
        y_imag = y[..., 1]
        if iteration > 0:
            padding = max(x.size(2) % 2, x.size(3) % 2)
            x_real = F.pad(x_real, pad=[padding, 0, padding, 0], mode='replicate')
            x_imag = F.pad(x_imag, pad=[padding, 0, padding, 0], mode='replicate')
            y_real = F.pad(y_real, pad=[padding, 0, padding, 0], mode='replicate')
            y_imag = F.pad(y_imag, pad=[padding, 0, padding, 0], mode='replicate')

            x_real = F.avg_pool2d(x_real, kernel_size=2, padding=0)
            x_imag = F.avg_pool2d(x_imag, kernel_size=2, padding=0)
            y_real = F.avg_pool2d(y_real, kernel_size=2, padding=0)
            y_imag = F.avg_pool2d(y_imag, kernel_size=2, padding=0)
            x = torch.stack((x_real, x_imag), dim=-1)
            y = torch.stack((y_real, y_imag), dim=-1)

        ssim_val, cs = _ssim_per_channel_complex(x, y, kernel=kernel, data_range=data_range, k1=k1, k2=k2)
        mcs.append(cs)

    # mcs, (level, batch)
    mcs_ssim = torch.relu(torch.stack(mcs[:-1] + [ssim_val], dim=0))

    mcs_ssim_real = mcs_ssim[..., 0]
    mcs_ssim_imag = mcs_ssim[..., 1]
    mcs_ssim_abs = (mcs_ssim_real.pow(2) + mcs_ssim_imag.pow(2)).sqrt()
    mcs_ssim_deg = torch.atan2(mcs_ssim_imag, mcs_ssim_real)

    mcs_ssim_pow_abs = mcs_ssim_abs ** scale_weights_tensor.view(-1, 1, 1)
    mcs_ssim_pow_deg = mcs_ssim_deg * scale_weights_tensor.view(-1, 1, 1)

    msssim_val_abs = torch.prod(mcs_ssim_pow_abs, dim=0)
    msssim_val_deg = torch.sum(mcs_ssim_pow_deg, dim=0)
    msssim_val_real = msssim_val_abs * torch.cos(msssim_val_deg)
    msssim_val_imag = msssim_val_abs * torch.sin(msssim_val_deg)
    msssim_val = torch.stack((msssim_val_real, msssim_val_imag), dim=-1).mean(dim=1)
    return msssim_val
