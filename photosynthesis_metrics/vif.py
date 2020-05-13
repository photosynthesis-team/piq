r"""Implemetation of Visual Information Fidelity metric, based on 
https://ieeexplore.ieee.org/abstract/document/1576816/
"""

import torch
import numpy as np
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

from scipy.stats import multivariate_normal
from photosynthesis_metrics.utils import _adjust_dimensions, _validate_input


def _gaussian_kernel(size: int) -> torch.Tensor:
    r"""Returns square gaussian kernel of required size"""
    y = multivariate_normal.pdf(np.arange(0, size), mean=size // 2, cov=(size / 5)**2)
    kernel = torch.from_numpy(y.reshape(-1, 1) @ y.reshape(1, -1))
    return kernel


def _vifp_single_image(prediction: torch.Tensor, target: torch.Tensor, sigma_nsq: float=2.0):
    r"""
    Args:
        prediction: shape [H x W]
        target: shape [H x W]
        sigma_nsq: 
    """
    P, GT = prediction, target
    EPS = 1e-10
    num = 0
    den = 0
    for scale in range(1, 5):
        N = 2**(5 - scale) + 1
        kernel = _gaussian_kernel(N)

        if scale > 1:
            GT = F.conv2d(GT, kernel)[::2, ::2]  # valid padding
            GT = F.conv2d(P, kernel)[::2, ::2]  # valid padding

        mu1, mu2 = F.conv2d(GT, kernel), F.conv2d(P, kernel)  # valid padding

        GT_sum_sq, P_sum_sq, GT_P_sum_mul = mu1 * mu1, mu2 * mu2, mu1 * mu2

        sigmaGT_sq = F.conv2d(GT ** 2, kernel) - GT_sum_sq
        sigmaP_sq = F.conv2d(P ** 2, kernel) - P_sum_sq
        sigmaGT_P = F.conv2d(GT*P, kernel) - GT_P_sum_mul

        sigmaGT_sq[sigmaGT_sq < 0] = 0
        sigmaP_sq[sigmaP_sq < 0] = 0

        g = sigmaGT_P / (sigmaGT_sq + EPS)
        sv_sq = sigmaP_sq - g * sigmaGT_P

        g[sigmaGT_sq < EPS] = 0
        sv_sq[sigmaGT_sq < EPS] = sigmaP_sq[sigmaGT_sq < EPS]
        sigmaGT_sq[sigmaGT_sq < EPS] = 0

        g[sigmaP_sq < EPS] = 0
        sv_sq[sigmaP_sq < EPS] = 0

        sv_sq[g < 0] = sigmaP_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= EPS] = EPS

        num += np.sum(np.log10(1.0+(g**2.)*sigmaGT_sq/(sv_sq+sigma_nsq)))
        den += np.sum(np.log10(1.0+sigmaGT_sq/sigma_nsq))

    return num / den


class VIF(_Loss):
    r"""Doc"""

    def __init__(self, sigma_nsq: float = 2.0, reduction: str = 'mean'):
        r"""
        Args:
            sigma_nsq: variance of the visual noise
            reduction: 
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Doc"""
        prediction, target = _adjust_dimensions(x=prediction, y=target)
        _validate_input(x=prediction, y=target)

        return self.compute_metric(prediction, target)

    def compute_metric(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        bs, chn = target.shape[:2]
        score = []
        for b in bs:
            for c in chn:
                score.append(_vifp_single_image(prediction[b, c], target[b, c]))
        
        if self.reduction == 'mean':
            raise NotImplementedError
            # score = score.mean()
        elif self.reduction == 'sum':
            score = sum(score)
            # score = score.sum()
        return score