import torch

import numpy as np

from scipy import linalg
from typing import Tuple, Optional
from torch import nn

from photosynthesis_metrics.feature_extractors.fid_inception import InceptionV3


def frechet_inception_distance(predicted_images: torch.Tensor, target_images: torch.Tensor,
                               feature_extractor: Optional[nn.Module] = None) -> float:
    assert isinstance(predicted_images, torch.Tensor) and isinstance(target_images, torch.Tensor), \
        f'Both image stacks must be of type torch.Tensor, got {type(predicted_images)} and {type(target_images)}.'

    # There is no assert for the full equality of shapes because it is not required.
    # Stacks can have any number of elements, but shapes of feature maps obtained from the images need to be equal.
    # Otherwise it will not be possible to correctly compute statistics.
    predicted_image_shape, target_image_shape = predicted_images.shape[1:], target_images.shape[1:]
    assert predicted_image_shape == target_image_shape, \
        f'Both image stacks must have images of the same shape, got {predicted_image_shape} and {target_image_shape}.'
    assert isinstance(feature_extractor, nn.Module) or feature_extractor is None, \
        f'Only PyTorch models are supported as feature extractors, got {type(feature_extractor)}.'

    if feature_extractor is None:
        print('WARNING: default feature extractor (InceptionNet V2) is used.')
        feature_extractor = InceptionV3()

    predicted_features = feature_extractor(predicted_images)
    target_features = feature_extractor(target_images)

    # TODO: `compute_fid` works with np.arrays, but need to work with torch.Tensors. Refactor that
    return compute_fid(predicted_features, target_features)


def __compute_fid(mu1: float, sigma1: np.ndarray, mu2: float, sigma2: np.ndarray, eps=1e-6) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

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


def compute_fid(gt_stack: np.ndarray, denoised_stack: np.ndarray) -> float:
    m_gt, s_gt = compute_statistics(gt_stack)
    m_denoised, s_denoised = compute_statistics(denoised_stack)

    fid_gt_denoised = __compute_fid(m_gt, s_gt, m_denoised, s_denoised)
    return fid_gt_denoised


def compute_statistics(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(stack, axis=0)
    sigma = np.cov(stack, rowvar=False)
    return mu, sigma
