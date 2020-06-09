r"""Implemetation of Visual Information Fidelity metric
Code is based on MATLAB version for computations in pixel domain
https://live.ece.utexas.edu/research/Quality/VIF.htm

References:
    https://ieeexplore.ieee.org/abstract/document/1576816/
"""
import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from typing import Union

from photosynthesis_metrics.utils import _adjust_dimensions, _validate_input


def _gaussian_kernel2d(kernel_size: int = 5, sigma: float = 2.0) -> torch.Tensor:
    r"""Returns 2D Gaussian kernel N(0,`sigma`)
    Args:
        kernel_size: Size
        sigma: Sigma
    Returns:
        gaussian_kernel: 2D kernel with shape (kernel_size x kernel_size)
        
    """
    x = torch.arange(- (kernel_size // 2), kernel_size // 2 + 1).resize(1, kernel_size)
    y = torch.arange(- (kernel_size // 2), kernel_size // 2 + 1).resize(kernel_size, 1)
    kernel = torch.exp(-(x * x + y * y) / (2.0 * sigma ** 2))
    # Normalize
    kernel = kernel / torch.sum(kernel)
    return kernel


def vif_p(prediction: torch.Tensor, target: torch.Tensor,
          sigma_n_sq: float = 2.0, data_range: Union[int, float] = 1.0) -> torch.Tensor:
    r"""Compute Visiual Information Fidelity in **pixel** domain for a batch of images.
    This metric isn't symmetric, so make sure to place arguments in correct order.

    Both inputs supposed to be in range [0, 1] with RGB order.
    Args:
        prediction: Batch of predicted images with shape (batch_size x channels x H x W)
        target: Batch of target images with shape  (batch_size x channels x H x W)
        sigma_n_sq: HVS model parameter (variance of the visual noise).
        data_range: Value range of input images (usually 1.0 or 255). Default: 1.0
        
    Returns:
        VIF: Index of similarity betwen two images. Usually in [0, 1] interval.
            Can be bigger than 1 for predicted images with higher contrast than original one.
    Note:
        In original paper this method was used for bands in discrete wavelet decomposition.
        Later on authors released code to compute VIF approximation in pixel domain.
        See https://live.ece.utexas.edu/research/Quality/VIF.htm for details.
        
    """
    assert prediction.dim() == 4, f'Expected 4D tensor, got {prediction.size()}'
    if data_range == 255:
        prediction = prediction / 255.
        target = target / 255.

    # Convert RGB image to YCbCr and take luminance: Y = 0.299 R + 0.587 G + 0.114 B
    num_channels = prediction.size(1)
    if num_channels == 3:
        prediction = 0.299 * prediction[:, 0, :, :] + 0.587 * prediction[:, 1, :, :] + 0.114 * prediction[:, 2, :, :]
        target = 0.299 * target[:, 0, :, :] + 0.587 * target[:, 1, :, :] + 0.114 * target[:, 2, :, :]

        # Add channel dimension
        prediction = prediction[:, None, :, :]
        target = target[:, None, :, :]
    
    # Constant for numerical stability
    EPS = 1e-10
    
    # Progressively downsample images and compute VIF on different scales
    prediction_vif, target_vif = 0, 0
    for scale in range(1, 5):
        kernel_size = 2 ** (5 - scale) + 1
        kernel = _gaussian_kernel2d(kernel_size, sigma=kernel_size / 5)
        kernel = kernel.view(1, 1, kernel_size, kernel_size).to(prediction)

        if scale > 1:
            # Convolve and downsample
            prediction = F.conv2d(prediction, kernel)[:, :, ::2, ::2]  # valid padding
            target = F.conv2d(target, kernel)[:, :, ::2, ::2]  # valid padding

        mu_trgt, mu_pred = F.conv2d(target, kernel), F.conv2d(prediction, kernel)  # valid padding
        mu_trgt_sq, mu_pred_sq, mu_trgt_pred = mu_trgt * mu_trgt, mu_pred * mu_pred, mu_trgt * mu_pred

        sigma_trgt_sq = F.conv2d(target ** 2, kernel) - mu_trgt_sq
        sigma_pred_sq = F.conv2d(prediction ** 2, kernel) - mu_pred_sq
        sigma_trgt_pred = F.conv2d(target * prediction, kernel) - mu_trgt_pred
        
        # Zero small negative values
        torch.relu_(sigma_trgt_sq)
        torch.relu_(sigma_pred_sq)
        
        g = sigma_trgt_pred / (sigma_trgt_sq + EPS)
        sigma_v_sq = sigma_pred_sq - g * sigma_trgt_pred

        pred_vif_scale = torch.log10(1.0 + (g ** 2.) * sigma_trgt_sq / (sigma_v_sq + sigma_n_sq))
        prediction_vif += torch.sum(pred_vif_scale, dim=[1, 2, 3])
        target_vif += torch.sum(torch.log10(1.0 + sigma_trgt_sq / sigma_n_sq), dim=[1, 2, 3])

    return prediction_vif / target_vif


class VIFLoss(_Loss):
    r"""Creates a criterion that measures the Visual Information Fidelity loss
    between predicted and target image. In order to be considered as a loss,
    value `1 - clip(VIF, min=0, max=1)` is returned.
    """

    def __init__(self, sigma_n_sq: float = 2.0, data_range: Union[int, float] = 1.0):
        r"""
        Args:
            sigma_n_sq: HVS model parameter (variance of the visual noise).
            data_range: Value range of input images (usually 1.0 or 255). Default: 1.0
        """
        super().__init__()
        self.sigma_n_sq = sigma_n_sq
        self.data_range = data_range

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Computation of Visual Information Fidelity (VIF) index as a loss function.
        Args:
            prediction: Tensor of prediction of the network.
            target: Reference tensor.
        Returns:
            Value of VIF loss to be minimized. 0 <= VIFLoss <= 1.
        """
        _validate_input(input_tensors=(prediction, target), allow_5d=False)
        prediction, target = _adjust_dimensions(input_tensors=(prediction, target))

        return self.compute_metric(prediction, target)

    def compute_metric(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        score = vif_p(prediction, target, sigma_n_sq=self.sigma_n_sq, data_range=self.data_range)
        # Make sure value to be in [0, 1] range and convert to loss
        loss = 1 - torch.clamp(torch.mean(score, dim=0), 0, 1)
        return loss
