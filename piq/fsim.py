r"""Implemetation of Feature Similarity Index Measure
Code is based on MATLAB version for computations in pixel domain
https://www4.comp.polyu.edu.hk/~cslzhang/IQA/FSIM/Files/FeatureSIM.m
References:
    https://www4.comp.polyu.edu.hk/~cslzhang/IQA/TIP_IQA_FSIM.pdf
"""
import math
import functools
from typing import Union, Tuple

import torch
from torch.nn.modules.loss import _Loss

from piq.utils import _adjust_dimensions, _validate_input
from piq.functional import ifftshift, get_meshgrid, similarity_map, gradient_map, scharr_filter, rgb2yiq


def fsim(x: torch.Tensor, y: torch.Tensor, reduction: str = 'mean',
         data_range: Union[int, float] = 1.0, chromatic: bool = True,
         scales: int = 4, orientations: int = 4, min_length: int = 6,
         mult: int = 2, sigma_f: float = 0.55, delta_theta: float = 1.2,
         k: float = 2.0) -> torch.Tensor:
    r"""Compute Feature Similarity Index Measure for a batch of images.
    

    Args:
        x: Batch of predicted images with shape (batch_size x channels x H x W)
        y: Batch of target images with shape  (batch_size x channels x H x W)
        data_range: Value range of input images (usually 1.0 or 255). Default: 1.0
        chromatic: Flag to compute FSIMc, which also takes into account chromatic components
        scales: Number of wavelets used for computation of phase congruensy maps
        orientations: Number of filter orientations used for computation of phase congruensy maps
        min_length: Wavelength of smallest scale filter
        mult: Scaling factor between successive filters
        sigma_f: Ratio of the standard deviation of the Gaussian describing the log Gabor filter's
            transfer function in the frequency domain to the filter center frequency.
        delta_theta: Ratio of angular interval between filter orientations and the standard deviation
            of the angular Gaussian function used to construct filters in the frequency plane.
        k: No of standard deviations of the noise energy beyond the mean at which we set the noise
            threshold  point, below which phase congruency values get penalized.
        
    Returns:
        FSIM: Index of similarity betwen two images. Usually in [0, 1] interval.
            Can be bigger than 1 for predicted images with higher contrast than the original ones.
    Note:
        This implementation is based on the original MATLAB code.
        https://www4.comp.polyu.edu.hk/~cslzhang/IQA/FSIM/FSIM.htm
        
    """
    
    _validate_input(input_tensors=(x, y), allow_5d=False)
    x, y = _adjust_dimensions(input_tensors=(x, y))
    
    # Scale to [0., 1.] range
    x = x / float(data_range)
    y = y / float(data_range)
    
    # Rescale to [0, 255] range, because all constant are calculated for this factor
    x = x * 255
    y = y * 255
    
    # Apply average pooling
    kernel_size = max(1, round(min(x.shape[-2:]) / 256))
    x = torch.nn.functional.avg_pool2d(x, kernel_size)
    y = torch.nn.functional.avg_pool2d(y, kernel_size)
        
    num_channels = x.size(1)

    # Convert RGB to YIQ color space https://en.wikipedia.org/wiki/YIQ
    if num_channels == 3:
        x_yiq = rgb2yiq(x)
        y_yiq = rgb2yiq(y)

        x_lum = x_yiq[:, : 1]
        y_lum = y_yiq[:, : 1]
        
        x_i = x_yiq[:, 1:2]
        y_i = y_yiq[:, 1:2]
        x_q = x_yiq[:, 2:]
        y_q = y_yiq[:, 2:]

    else:
        x_lum = x
        y_lum = y

    # Compute phase congruency maps
    pc_x = _phase_congruency(
        x_lum, scales=scales, orientations=orientations,
        min_length=min_length, mult=mult, sigma_f=sigma_f,
        delta_theta=delta_theta, k=k
    )
    pc_y = _phase_congruency(
        y_lum, scales=scales, orientations=orientations,
        min_length=min_length, mult=mult, sigma_f=sigma_f,
        delta_theta=delta_theta, k=k
    )
    
    # Gradient maps
    kernels = torch.stack([scharr_filter(), scharr_filter().transpose(-1, -2)])
    grad_map_x = gradient_map(x_lum, kernels)
    grad_map_y = gradient_map(y_lum, kernels)
    
    # Constants from the paper
    T1, T2, T3, T4, lmbda = 0.85, 160, 200, 200, 0.03
    
    # Compute FSIM
    PC = similarity_map(pc_x, pc_y, T1)
    GM = similarity_map(grad_map_x, grad_map_y, T2)
    pc_max = torch.where(pc_x > pc_y, pc_x, pc_y)
    score = GM * PC * pc_max
    
    if chromatic:
        S_I = similarity_map(x_i, y_i, T3)
        S_Q = similarity_map(x_q, y_q, T4)
        score = score * torch.abs(S_I * S_Q) ** lmbda
        # Complex gradients will work in PyTorch 1.6.0
        # score = score * torch.real((S_I * S_Q).to(torch.complex64) ** lmbda)

    result = score.sum(dim=[1, 2, 3]) / pc_max.sum(dim=[1, 2, 3])
    
    if reduction == 'none':
        return result

    return {'mean': result.mean,
            'sum': result.sum
            }[reduction](dim=0)


def _construct_filters(x: torch.Tensor, scales: int = 4, orientations: int = 4,
                       min_length: int = 6, mult: int = 2, sigma_f: float = 0.55,
                       delta_theta: float = 1.2, k: float = 2.0):
    """Creates a stack of filters used for computation of phase congruensy maps
    
    Args:
        x: Tensor with shape Bx1xHxW
        scales: Number of wavelets
        orientations: Number of filter orientations
        min_length: Wavelength of smallest scale filter
        mult: Scaling factor between successive filters
        sigma_f: Ratio of the standard deviation of the Gaussian
            describing the log Gabor filter's transfer function
            in the frequency domain to the filter center frequency.
        delta_theta: Ratio of angular interval between filter orientations
            and the standard deviation of the angular Gaussian function
            used to construct filters in the freq. plane.
        k: No of standard deviations of the noise energy beyond the mean
            at which we set the noise threshold point, below which phase
            congruency values get penalized.
        """
    B, _, H, W = x.shape

    # Calculate the standard deviation of the angular Gaussian function
    # used to construct filters in the freq. plane.
    theta_sigma = math.pi / (orientations * delta_theta)

    # Pre-compute some stuff to speed up filter construction
    grid_x, grid_y = get_meshgrid((H, W))
    radius = torch.sqrt(grid_x ** 2 + grid_y ** 2)
    theta = torch.atan2(-grid_y, grid_x)

    # Quadrant shift radius and theta so that filters are constructed with 0 frequency at the corners.
    # Get rid of the 0 radius value at the 0 frequency point (now at top-left corner)
    # so that taking the log of the radius will not cause trouble.
    radius = ifftshift(radius)
    theta = ifftshift(theta)
    radius[0, 0] = 1

    sintheta = torch.sin(theta)
    costheta = torch.cos(theta)

    # Filters are constructed in terms of two components.
    # 1) The radial component, which controls the frequency band that the filter responds to
    # 2) The angular component, which controls the orientation that the filter responds to.
    # The two components are multiplied together to construct the overall filter.

    # First construct a low-pass filter that is as large as possible, yet falls
    # away to zero at the boundaries.  All log Gabor filters are multiplied by
    # this to ensure no extra frequencies at the 'corners' of the FFT are
    # incorporated as this seems to upset the normalisation process when
    lp = _lowpassfilter(size=(H, W), cutoff=.45, n=15)

    # Construct the radial filter components...
    log_gabor = []
    for s in range(scales):
        wavelength = min_length * mult ** s
        omega_0 = 1.0 / wavelength
        gabor_filter = torch.exp((- torch.log(radius / omega_0) ** 2) / (2 * math.log(sigma_f) ** 2))
        gabor_filter = gabor_filter * lp
        gabor_filter[0, 0] = 0
        log_gabor.append(gabor_filter)

    # Then construct the angular filter components...
    spread = []
    for o in range(orientations):
        angl = o * math.pi / orientations

        # For each point in the filter matrix calculate the angular distance from
        # the specified filter orientation.  To overcome the angular wrap-around
        # problem sine difference and cosine difference values are first computed
        # and then the atan2 function is used to determine angular distance.
        ds = sintheta * math.cos(angl) - costheta * math.sin(angl)  # Difference in sine.
        dc = costheta * math.cos(angl) + sintheta * math.sin(angl)  # Difference in cosine.
        dtheta = torch.abs(torch.atan2(ds, dc))
        spread.append(torch.exp((- dtheta ** 2) / (2 * theta_sigma ** 2)))

    spread = torch.stack(spread)
    log_gabor = torch.stack(log_gabor)
    
    # Multiply, add batch dimension and transfer to correct device.
    filters = (spread.repeat_interleave(scales, dim=0) * log_gabor.repeat(orientations, 1, 1)).unsqueeze(0).to(x)
    return filters


def _phase_congruency(x: torch.Tensor, scales: int = 4, orientations: int = 4,
                      min_length: int = 6, mult: int = 2, sigma_f: float = 0.55,
                      delta_theta: float = 1.2, k: float = 2.0) -> torch.Tensor:
    r"""Compute Phase Congruence for a batch of greyscale images

    Args:
        x: Tensor with shape Bx1xHxW
        levels: Number of wavelet scales
        orientations: Number of filter orientations
        min_length: Wavelength of smallest scale filter
        mult: Scaling factor between successive filters
        sigma_f: Ratio of the standard deviation of the Gaussian
            describing the log Gabor filter's transfer function
            in the frequency domain to the filter center frequency.
        delta_theta: Ratio of angular interval between filter orientations
            and the standard deviation of the angular Gaussian function
            used to construct filters in the freq. plane.
        k: No of standard deviations of the noise energy beyond the mean
            at which we set the noise threshold point, below which phase
            congruency values get penalized.
    Returns:
        PCmap: Tensor with shape BxHxW

    """
    EPS = torch.finfo(x.dtype).eps

    B, _, H, W = x.shape

    # Fourier transform
    imagefft = torch.rfft(x, 2, onesided=False)

    filters = _construct_filters(x, scales, orientations, min_length, mult, sigma_f, delta_theta, k)

    # Note rescaling to match power record ifft2 of filter
    filters_ifft = torch.ifft(torch.stack([filters, torch.zeros_like(filters)], dim=-1), 2)[..., 0] * math.sqrt(H * W)
    
    # Convolve image with even and odd filters
    even_odd = torch.ifft(imagefft * filters.unsqueeze(-1), 2).view(B, orientations, scales, H, W, 2)

    # Amplitude of even & odd filter response. An = sqrt(real^2 + imag^2)
    an = torch.sqrt(torch.sum(even_odd ** 2, dim=-1))

    # Take filter at scale 0 and sum spatially
    # Record mean squared filter value at smallest scale.
    # This is used for noise estimation.
    em_n = (filters.view(1, orientations, scales, H, W)[:, :, :1, ...] ** 2).sum(dim=[-2, -1], keepdims=True)

    # Sum of even filter convolution results.
    sum_e = even_odd[..., 0].sum(dim=2, keepdims=True)
    
    # Sum of odd filter convolution results.
    sum_o = even_odd[..., 1].sum(dim=2, keepdims=True)
    
    # Get weighted mean filter response vector, this gives the weighted mean phase angle.
    x_energy = torch.sqrt(sum_e ** 2 + sum_o ** 2) + EPS

    mean_e = sum_e / x_energy
    mean_o = sum_o / x_energy

    # Now calculate An(cos(phase_deviation) - | sin(phase_deviation)) | by
    # using dot and cross products between the weighted mean filter response
    # vector and the individual filter response vectors at each scale.
    # This quantity is phase congruency multiplied by An, which we call energy.

    # Extract even and odd convolution results.
    even = even_odd[..., 0]
    odd = even_odd[..., 1]

    energy = (even * mean_e + odd * mean_o - torch.abs(even * mean_o - odd * mean_e)).sum(dim=2, keepdim=True)
    
    # Compensate for noise
    # We estimate the noise power from the energy squared response at the
    # smallest scale.  If the noise is Gaussian the energy squared will have a
    # Chi-squared 2DOF pdf.  We calculate the median energy squared response
    # as this is a robust statistic.  From this we estimate the mean.
    # The estimate of noise power is obtained by dividing the mean squared
    # energy value by the mean squared filter value
    
    abs_eo = torch.sqrt(torch.sum(even_odd[:, :, :1, ...] ** 2, dim=-1)).reshape(B, orientations, 1, 1, H * W)
    median_e2n = torch.median(abs_eo ** 2, dim=-1, keepdims=True).values

    mean_e2n = - median_e2n / math.log(0.5)

    # Estimate of noise power.
    noise_power = mean_e2n / em_n
    
    # Now estimate the total energy^2 due to noise
    # Estimate for sum(An^2) + sum(Ai.*Aj.*(cphi.*cphj + sphi.*sphj))
    filters_ifft = filters_ifft.view(1, orientations, scales, H, W)
    
    sum_an2 = torch.sum(filters_ifft ** 2, dim=-3, keepdim=True)
    
    sum_ai_aj = torch.zeros(B, orientations, 1, H, W).to(x)
    for s in range(scales - 1):
        sum_ai_aj = sum_ai_aj + (filters_ifft[:, :, s: s + 1] * filters_ifft[:, :, s + 1:]).sum(dim=-3, keepdim=True)
            
    sum_an2 = torch.sum(sum_an2, dim=[-1, -2], keepdim=True)
    sum_ai_aj = torch.sum(sum_ai_aj, dim=[-1, -2], keepdim=True)

    noise_energy2 = 2 * noise_power * sum_an2 + 4 * noise_power * sum_ai_aj

    # Rayleigh parameter
    tau = torch.sqrt(noise_energy2 / 2)

    # Expected value of noise energy
    noise_energy = tau * math.sqrt(math.pi / 2)
    moise_energy_sigma = torch.sqrt((2 - math.pi / 2) * tau ** 2)

    # Noise threshold
    T = noise_energy + k * moise_energy_sigma

    # The estimated noise effect calculated above is only valid for the PC_1 measure.
    # The PC_2 measure does not lend itself readily to the same analysis.  However
    # empirically it seems that the noise effect is overestimated roughly by a factor
    # of 1.7 for the filter parameters used here.

    # Empirical rescaling of the estimated noise effect to suit the PC_2 phase congruency measure
    T = T / 1.7

    # Apply noise threshold
    energy = torch.max(energy - T, torch.zeros_like(T))

    eps = torch.finfo(energy.dtype).eps
    energy_all = energy.sum(dim=[1, 2]) + eps
    an_all = an.sum(dim=[1, 2]) + eps
    result_pc = energy_all / an_all
    return result_pc.unsqueeze(1)


def _lowpassfilter(size: Union[int, Tuple[int, int]], cutoff: float, n: int) -> torch.Tensor:
    r"""
    Constructs a low-pass Butterworth filter.
    Args:
        size: Tuple with heigth and width of filter to construct
        cutoff: Cutoff frequency of the filter in (0, 0.5()
        n: Filter order. Higher `n` means sharper transition.
            Note that `n` is doubled so that it is always an even integer.
        
    Returns:
        f = 1 / (1 + w/cutoff) ^ 2n
    
    Note:
        The frequency origin of the returned filter is at the corners.
    
    """
    assert 0 < cutoff <= 0.5, "Cutoff frequency must be between 0 and 0.5"
    assert n > 1 and int(n) == n, "n must be an integer >= 1"

    grid_x, grid_y = get_meshgrid(size)
    
    # A matrix with every pixel = radius relative to centre.
    radius = torch.sqrt(grid_x ** 2 + grid_y ** 2)

    return ifftshift(1. / (1.0 + (radius / cutoff) ** (2 * n)))


class FSIMLoss(_Loss):
    r"""Creates a criterion that measures the FSIM or FSIMc for input :math:`x` and target :math:`y`.

    In order to be considered as a loss, value `1 - clip(FSIM, min=0, max=1)` is returned. If you need FSIM value,
    use function `fsim` instead.

    Args:
        chromatic: Flag to compute FSIMc, which also takes into account chromatic components
        reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        data_range: The difference between the maximum and minimum of the pixel value,
            i.e., if for image x it holds min(x) = 0 and max(x) = 1, then data_range = 1.
            The pixel value interval of both input and output should remain the same.
        scales: Number of wavelets used for computation of phase congruensy maps
        orientations: Number of filter orientations used for computation of phase congruensy maps
        min_length: Wavelength of smallest scale filter
        mult: Scaling factor between successive filters
        sigma_f: Ratio of the standard deviation of the Gaussian describing the log Gabor filter's
            transfer function in the frequency domain to the filter center frequency.
        delta_theta: Ratio of angular interval between filter orientations and the standard deviation
            of the angular Gaussian function used to construct filters in the frequency plane.
        k: No of standard deviations of the noise energy beyond the mean at which we set the noise
            threshold  point, below which phase congruency values get penalized.

    Shape:
        - Input: Required to be 2D (H, W), 3D (C,H,W), 4D (N,C,H,W) or 5D (N,C,H,W,2), channels first.
        - Target: Required to be 2D (H, W), 3D (C,H,W), 4D (N,C,H,W) or 5D (N,C,H,W,2), channels first.

    Examples::

        >>> loss = FSIMLoss()
        >>> prediction = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> target = torch.rand(3, 3, 256, 256)
        >>> output = loss(prediction, target)
        >>> output.backward()

    References:
        .. [1] Anish Mittal et al. "No-Reference Image Quality Assessment in the Spatial Domain",
        https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf
        """
    def __init__(self, data_range: Union[int, float] = 1., reduction: str = 'mean', scales: int = 4,
                 orientations: int = 4, min_length: int = 6, mult: int = 2, sigma_f: float = 0.55,
                 delta_theta: float = 1.2, k: float = 2.0) -> None:

        super().__init__()
        self.data_range = data_range
        self.reduction = reduction

        # Save function with predefined parameters, rather than parameters themself
        self.fsim = functools.partial(
            fsim,
            data_range=data_range,
            reduction=reduction,
            scales=scales,
            orientations=orientations,
            min_length=min_length,
            mult=mult,
            sigma_f=sigma_f,
            delta_theta=delta_theta,
            k=k,
        )

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Computation of FSIM as a loss function.
        Args:
            prediction: Tensor of prediction of the network.
            target: Reference tensor.
        Returns:
            Value of FSIM loss to be minimized. 0 <= FSIM <= 1.
        """
        # All checks are done inside fsim function
        score = self.fsim(prediction, target)

        # Make sure value to be in [0, 1] range and convert to loss
        return 1 - torch.clamp(score, 0, 1)
