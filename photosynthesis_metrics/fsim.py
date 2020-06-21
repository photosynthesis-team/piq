import math

import torch
from typing import Union, Tuple

from photosynthesis_metrics.utils import _adjust_dimensions, _validate_input


def fsim(x: torch.Tensor, y: torch.Tensor,
         reduction: str = 'mean',
         data_range: Union[int, float] = 1.0,
         chromatic: bool = True) -> torch.Tensor:
    r"""Compute Feature Similarity Index Measure for a batch of images.
    

    Args:
        x: Batch of predicted images with shape (batch_size x channels x H x W)
        y: Batch of target images with shape  (batch_size x channels x H x W)

        data_range: Value range of input images (usually 1.0 or 255). Default: 1.0
        chromatic: Flag to compute FSIMc, which also takes into account chromatic components
        
    Returns:
        FSIM: Index of similarity betwen two images. Usually in [0, 1] interval.
            Can be bigger than 1 for predicted images with higher contrast than original one.
    Note:
        This implementation is based on original authors MATLAB code.
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
    kernel_size = max(1, min(x.shape[-2:]) // 256)
    x = torch.nn.functional.avg_pool2d(x, kernel_size, stride=2)
    y = torch.nn.functional.avg_pool2d(y, kernel_size, stride=2)
        
    num_channels = x.size(1)

    # Convert RGB to YIQ color space https://en.wikipedia.org/wiki/YIQ
    if num_channels == 3:
        yiq_weights = torch.tensor([
            [0.229, 0.587, 0.114],
            [0.5959, -0.2746, -0.3213],
            [0.2115, -0.5227, 0.3112]]).t().to(x)
        x = torch.matmul(x.permute(0, 2, 3, 1), yiq_weights).permute(0, 3, 1, 2)
        y = torch.matmul(y.permute(0, 2, 3, 1), yiq_weights).permute(0, 3, 1, 2)
        
        x_Y = x[:, : 1, :, :]
        y_Y = y[:, : 1, :, :]
        
        x_I = x[:, 1: 2, :, :]
        y_I = y[:, 1: 2, :, :]
        x_Q = x[:, 2:, :, :]
        y_Q = y[:, 2:, :, :]

    else:
        x_Y = x
        y_Y = y

    # Compute phase congruency maps
    PCx = _phase_congruency(x_Y)
    PCy = _phase_congruency(y_Y)
    
    # Gradient maps
    kernel = torch.stack([_scharr_filter(), _scharr_filter().transpose(-1, -2)]).to(x)
    grad_x = torch.nn.functional.conv2d(x_Y, kernel, padding=1)
    grad_y = torch.nn.functional.conv2d(y_Y, kernel, padding=1)
    
    grad_map_x = torch.sqrt(torch.sum(grad_x ** 2, dim=-3, keepdim=True))
    grad_map_y = torch.sqrt(torch.sum(grad_y ** 2, dim=-3, keepdim=True))
    
    # Constants from paper
    T1, T2, T3, T4, lmbda = 0.85, 160, 200, 200, 0.03
    
    # Compute FSIM
    PC = (2.0 * PCx * PCy + T1) / (PCx ** 2 + PCy ** 2 + T1)
    GM = (2.0 * grad_map_x * grad_map_y + T2) / (grad_map_x ** 2 + grad_map_y ** 2 + T2)
    PCmax = torch.where(PCx > PCy, PCx, PCy)
    
    score = GM * PC * PCmax
    
    if chromatic:
        S_I = (2 * x_I * y_I + T3) / (x_I ** 2 + y_I ** 2 + T3)
        S_Q = (2 * x_Q * y_Q + T4) / (x_Q ** 2 + y_Q ** 2 + T4)
        print(score.shape, S_I.shape, S_Q.shape)
        score = score * (S_I * S_Q) ** lmbda

    result = score.sum(dim=[2, 3]) / PCmax.sum(dim=[2, 3])
    
    if reduction == 'none':
        return result

    return {'mean': result.mean,
            'sum': result.sum
            }[reduction](dim=0)


def _scharr_filter() -> torch.Tensor:
    """Utility function that returns a normalized 3x3 Scharr kernel in X direction
    Return:
        kernel: Tensor with shape 1x3x3"""
    kernel = torch.tensor([[[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]]]) / 16
    return kernel


def _construct_filters(x: torch.Tensor, scales: int = 4, orientations: int = 4,
                       min_length: int = 6, mult: int = 2, sigma_f: float = 0.55,
                       delta_theta: float = 1.2, k: float = 2.0):
    """Creates stack of filters used for computation of phase congruensy maps
    
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
    grid_x, grid_y = _get_meshgrid((H, W))
    radius = torch.sqrt(grid_x ** 2 + grid_y ** 2)
    theta = torch.atan2(-grid_y, grid_x).t()

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
    logGabor = []
    for s in range(scales):
        wavelength = min_length * mult ** s
        fo = 1.0 / wavelength  # Centre frequency of filter.
        gabor_filter = torch.exp((- torch.log(radius / fo) ** 2) / (2 * math.log(sigma_f) ** 2))
        gabor_filter = gabor_filter * lp
        gabor_filter[0, 0] = 0
        logGabor.append(gabor_filter)

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
    logGabor = torch.stack(logGabor)
    
    # Multiply, add batch dimension and transfer to correct device.
    filters = (spread.repeat_interleave(scales, dim=0) * logGabor.repeat(orientations, 1, 1)).unsqueeze(0).to(x)
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
    """
    EPS = 1e-4

    B, _, H, W = x.shape

    # Fourier transform
    imagefft = torch.rfft(x, 2, onesided=False)

    filters = _construct_filters(x, scales, orientations, min_length, mult, sigma_f, delta_theta, k)

    filters_ifft = torch.ifft(torch.stack([filters, torch.zeros_like(filters)], dim=-1), 2)[..., 0] * math.sqrt(H * W)
    
    # Convolve image with even and odd filters
    E0 = torch.ifft(imagefft * filters.unsqueeze(-1), 2).view(B, orientations, scales, H, W, 2)

    # Amplitude of even & odd filter response. An = sqrt(real^2 + imag^2)
    An = torch.sqrt(torch.sum(E0 ** 2, dim=-1))

    # Take filter at scale 0 and sum spatially
    # Record mean squared filter value at smallest scale.
    # This is used for noise estimation.
    EM_n = (filters.view(1, orientations, scales, H, W)[:, :, :1, ...] ** 2).sum(dim=[-2, -1], keepdims=True)
    
    # Sum of amplitude responses
    sumAn = An.sum(dim=2, keepdims=True)
    
    # Sum of even filter convolution results.
    sumE = E0[..., 0].sum(dim=2, keepdims=True)
    
    # Sum of odd filter convolution results.
    sumO = E0[..., 1].sum(dim=2, keepdims=True)
    
    # Get weighted mean filter response vector, this gives the weighted mean phase angle.
    XEnergy = torch.sqrt(sumE ** 2 + sumO ** 2) + EPS

    MeanE = sumE / XEnergy
    MeanO = sumO / XEnergy

    # Now calculate An(cos(phase_deviation) - | sin(phase_deviation)) | by
    # using dot and cross products between the weighted mean filter response
    # vector and the individual filter response vectors at each scale.
    # This quantity is phase congruency multiplied by An, which we call energy.

    # Extract even and odd convolution results.
    E = E0[..., 0]
    O = E0[..., 1]

    Energy = (E * MeanE + O * MeanO - torch.abs(E * MeanO - O * MeanE)).sum(dim=2, keepdim=True)
    
    # Compensate for noise
    # We estimate the noise power from the energy squared response at the
    # smallest scale.  If the noise is Gaussian the energy squared will have a
    # Chi-squared 2DOF pdf.  We calculate the median energy squared response
    # as this is a robust statistic.  From this we estimate the mean.
    # The estimate of noise power is obtained by dividing the mean squared
    # energy value by the mean squared filter value
    
    absE0 = torch.sqrt(torch.sum(E0[:, :, :1, ...] ** 2, dim=-1)).reshape(B, orientations, 1, 1, H * W)
    medianE2n = torch.median(absE0 ** 2, dim=-1, keepdims=True).values

    meanE2n = - medianE2n / math.log(0.5)

    # Estimate of noise power.
    noisePower = meanE2n / EM_n
    
    # Now estimate the total energy^2 due to noise
    # Estimate for sum(An^2) + sum(Ai.*Aj.*(cphi.*cphj + sphi.*sphj))
    filters_ifft = filters_ifft.view(1, orientations, scales, H, W)
    
    EstSumAn2 = torch.sum(filters_ifft ** 2, dim=-3, keepdim=True)
    
    EstSumAiAj = torch.zeros(B, orientations, 1, H, W).to(x)
    for s in range(scales - 1):
        EstSumAiAj = EstSumAiAj + (filters_ifft[:, :, s: s + 1] * filters_ifft[:, :, s + 1:]).sum(dim=-3, keepdim=True)
            
    sumEstSumAn2 = torch.sum(EstSumAn2, dim=[-1, -2], keepdim=True)
    sumEstSumAiAj = torch.sum(EstSumAiAj, dim=[-1, -2], keepdim=True)

    EstNoiseEnergy2 = 2 * noisePower * sumEstSumAn2 + 4 * noisePower * sumEstSumAiAj

    # Rayleigh parameter
    tau = torch.sqrt(EstNoiseEnergy2 / 2)

    # Expected value of noise energy
    EstNoiseEnergy = tau * math.sqrt(math.pi / 2)
    EstNoiseEnergySigma = torch.sqrt((2 - math.pi / 2) * tau ** 2)

    # Noise threshold
    T = EstNoiseEnergy + k * EstNoiseEnergySigma

    # The estimated noise effect calculated above is only valid for the PC_1 measure.
    # The PC_2 measure does not lend itself readily to the same analysis.  However
    # empirically it seems that the noise effect is overestimated roughly by a factor
    # of 1.7 for the filter parameters used here.

    # Empirical rescaling of the estimated noise effect to suit the PC_2 phase congruency measure
    T = T / 1.7

    # Apply noise threshold
    Energy = torch.max(Energy - T, torch.zeros_like(T))

    EnergyAll = Energy.sum(dim=1, keepdims=True)
    
    sumAn = An.sum(dim=2, keepdims=True)
    
    AnAll = sumAn.sum(dim=1, keepdims=True)
    
    ResultPC = (EnergyAll / AnAll).view(B, 1, H, W)
    return ResultPC


def _get_meshgrid(size):
    if size[0] % 2:
        # Odd
        x = torch.range(-(size[0] - 1) / 2, (size[0] - 1) / 2) / (size[0] - 1)
    else:
        # Even
        x = torch.range(- size[0] / 2, size[0] / 2 - 1) / size[0]
    
    if size[1] % 2:
        # Odd
        y = torch.range(-(size[1] - 1) / 2, (size[1] - 1) / 2) / (size[1] - 1)
    else:
        # Even
        y = torch.range(- size[1] / 2, size[1] / 2 - 1) / size[1]
    return torch.meshgrid(x, y)


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

    assert (cutoff >= 0) and (cutoff <= 0.5), "Cutoff frequency must be between 0 and 0.5"
    assert n > 1, "n must be an integer >= 1"

    grid_x, grid_y = _get_meshgrid(size)
    
    # A matrix with every pixel = radius relative to centre.
    radius = torch.sqrt(grid_x ** 2 + grid_y ** 2)

    return ifftshift(1. / (1.0 + (radius / cutoff) ** (2 * n)))


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    dim = tuple(range(x.dim()))
    shift = [(dim + 1) // 2 for dim in x.shape]

    return roll(x, shift, dim)


def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)
