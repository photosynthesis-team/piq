import numpy as np

from scipy import linalg

from typing import Tuple


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
