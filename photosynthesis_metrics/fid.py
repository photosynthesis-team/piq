"""
PyTorch implementation of Frechet Inception Distance (FID score)
Reference:
    Martin Heusel et al. "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", 
    https://arxiv.org/abs/1706.08500
Credits:
    https://github.com/hukkelas/pytorch-frechet-inception-distance/
    https://github.com/mseitzer/pytorch-fid
"""

from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from scipy import linalg
from torch import nn

from photosynthesis_metrics.feature_extractors.fid_inception import InceptionV3


def __compute_fid(mu1: float, sigma1: np.ndarray, mu2: float, sigma2: np.ndarray, eps=1e-6) -> float:
    r"""
    The Frechet distance between two multivariate Gaussians X_predicted ~ N(mu_1, sigm_1) 
    and X_target ~ N(mu_2, sigm_2) is
        d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).
    Args:
        mu1: mean of activations calculated on predicted samples
        sigma1: covariance matrix over activations calculated on predicted samples
        mu2: mean of activations calculated on target samples
        sigma2: covariance matrix over activations calculated on target samples
        eps: offset constant. used if sigma_1 @ sigma_2 matrix is singular
    """
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    # Product might be almost singular
    if not np.isfinite(covmean).all():
        print(f'fid calculation produces singular product; adding {eps} to diagonal of cov estimates')
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))

        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def compute_fid(predicted_stack: np.ndarray, target_stack: np.ndarray) -> float:
    r"""Numpy implementation of the Frechet Distance.
    Fits multivariate Gaussians: X_predicted ~ N(mu_1, sigm_1) and X_target ~ N(mu_2, sigm_2) to image stacks. 
    Then computes FID as d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).
    
    Args:
        predicted_stack: activations from encoder with shape (N_samples, dims) and dtype: np.float32 in range 0 - 1
        target_stack: activations from encoder with shape (N_samples, dims) and dtype: np.float32 in range 0 - 1
    Returns:
    --   : The Frechet Distance.
    """
    m_pred, s_pred = compute_statistics(predicted_stack)
    m_targ, s_targ = compute_statistics(target_stack)

    fid_score = __compute_fid(m_pred, s_pred, m_targ, s_targ)
    return fid_score


def compute_statistics(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""Calculates the statistics used by FID
    Args:
        stack: activations from encoder with shape (N_samples, dims) and dtype: np.float32 in range 0 - 1
    Returns:
        mu:     mean over all activations from the encoder.
        sigma:  covariance matrix over all activations from the encoder.
    """
    mu = np.mean(stack, axis=0)
    sigma = np.cov(stack, rowvar=False)
    return mu, sigma

class FID(nn.Module):
    r"""Creates a criterion that measures Frechet Inception Distance score for two datasets of images
    See https://arxiv.org/abs/1706.08500 for reference.
    """

    def __init__(self):
        super(FID, self).__init__()


    def forward(self, predicted_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        r"""Interface of Frechet Inception Distance.
        It's computed for a whole set of data and uses features from encoder instead of images itself to decrease computation cost.
        FID can compare two data distributions with different number of samples. But dimensionalities should match, otherwise 
        it won't be possible to correctly compute statistics.
    
        Args:
            predicted_features: Low-dimension representation of predicted image set. Shape (N_pred, encoder_dim)
            target_features: Low-dimension representation of target image set. Shape (N_targ, encoder_dim)
            
        Returns:
            normed_msidx: normalized msid descriptor (if no `target_generator` specified).
            msid_score: the scalar value of the distance between discriptor if both `prediction` and `target` given.
        """

        assert predicted_features.shape[1] == target_features.shape[1], \
            f"Features dimensionalities should match, otherwise it won't be possible to correctly compute statistics. \
                Got {predicted_features.shape[1]} and {target_features.shape[1]}"
        assert predicted_features.dims == 2, f"Predicted features must have shape  (N_samples, encoder_dim), got {predicted_features.dims}"
        assert target_features.dims == 2, f"Target features must have shape  (N_samples, encoder_dim), got {target_features.dims}"

        score = compute_fid(predicted_features.numpy(), target_features.numpy())
        return torch.Tensor(score)


    def _compute_feats(self, loader, feature_extractor : Callable = None, device: str = 'cuda') -> torch.Tensor:
        r"""Generate low-dimensional image desciptors to be used for computing MSID score. 
        Args:
            loader: Batch of images generator. Batche required to be 4D, channels first (N,C,H,W).
            feature_extractor: model used to generate image features, if None use `InceptionNet V2` model
            out_features: size of `feature_extractor` output
            device: Device on which to compute inference of the model
        """

        if feature_extractor is None:
            print('WARNING: default feature extractor (InceptionNet V2) is used.')
            feature_extractor = InceptionV3()
            # self.out_features = ... # Inception V2() out features
        else:
            assert callable(feature_extractor), f"Feature_extractor must be callabel. Got {type(feature_extractor)}"

        feature_extractor.eval()
        
        if out_features is None:
            # Figure `out_size` from output shape
            mock_batch = torch.rand((1, 3, 256, 256))
            out_features = feature_extractor(mock_batch).reshape((1, -1)).shape[1]

        total_feats = []
        for batch in loader:
            N = batch.shape[0]
            batch = batch.float().to(device)

            # Getting features
            batch_feats = feature_extractor(batch).view(N, )
            total_feats.append(batch_feats)
        return torch.Tensor(total_feats)