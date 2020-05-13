"""
PyTorch implementation of Inception score (IS)
Reference:
    Salimans T. et al. Improved techniques for training gans (2016)
    https://arxiv.org/abs/1606.03498
    Shane Barratt et al. A Note on the Inception Score
    https://arxiv.org/pdf/1801.01973.pdf

Credits:
    https://github.com/sbarratt/inception-score-pytorch
    https://github.com/tsc2017/Inception-Score
    https://github.com/openai/improved-gan/issues/29
"""
from typing import Optional

import torch
import torch.nn.functional as F

from photosynthesis_metrics.base import BaseFeatureMetric
from photosynthesis_metrics.utils import _validate_features

class IS(BaseFeatureMetric):
    r"""Creates a criterion that measures Inception Score.
    IS is computed separatly for predicted and target features and expects raw InceptionV3 model logits as inputs.

    Args:
        predicted_features (torch.Tensor): Low-dimension representation of predicted image set. Shape (N_pred, encoder_dim)
        target_features (torch.Tensor): Low-dimension representation of target image set. Shape (N_targ, encoder_dim)

    Returns:
        predicted_score: Scalar value of IS for predicted images features.
        target_score: Scalar value of IS for target images features if `ret_target` is True.

    Reference:
        https://arxiv.org/pdf/1801.01973.pdf
    """

    def __init__(self, num_splits: int = 10, ret_target: bool = False, ret_var: bool = False) -> None:
        r"""
        Args:
            num_splits: Number of parts to devide features. IS is computed for them separatly and results are then averaged.
            ret_target: If True, also return IS for target features. Default: False, to be compatible with other metrics
            ret_var: If True, returns mean and variance for each feature. Default: False.
        """
        super(IS, self).__init__()

        self.num_splits = num_splits
        self.ret_target = ret_target
        self.ret_var = ret_var

    def forward(
        self, predicted_features: torch.Tensor, target_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Sanity check for input.
        _validate_features(predicted_features, predicted_features if target_features is None else target_features)
        return self.compute_metric(predicted_features, target_features)

    def compute_metric(
        self, predicted_features: torch.Tensor, target_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Compute IS

        Returns:
            predicted_score: Scalar value of IS for predicted images features.
            target_score: Scalar value of IS for target images features if `ret_target` is True.
        """
        predicted_is = self.logits_to_score(predicted_features)
        if self.ret_target:
            target_is = self.logits_to_score(target_features)
            return predicted_is, target_is
        
        return predicted_is

    def logits_to_score(self, logits):
        r"""Computes mean KL divergence.
        """
        N = logits.size(0)
        probas = F.softmax(logits)
        split_scores = []
        for i in range(self.num_splits):
            part = probas[i * (N // self.num_splits): (i+1) * (N // self.num_splits), :]
            p_y = torch.mean(part, dim=0)
            scores = []
            for k in range(part.shape[0]):
                p_yx = part[k, :]
                scores.append(F.kl_div(p_y.log(), p_yx, reduction='sum'))
            # Compute exponential of the mean of the KL-divergence for each split
            split_scores.append(torch.tensor(scores).mean().exp())
        split_scores = torch.tensor(split_scores)
        if self.ret_var:
            # Move to the same device as input
            return torch.mean(split_scores).to(logits), torch.std(split_scores).to(logits)
        return torch.mean(split_scores).to(logits)


