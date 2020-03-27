from typing import Optional, List, Tuple, Union
from functools import partial

import numpy as np
import torch
from sklearn.metrics.pairwise import polynomial_kernel

from photosynthesis_metrics.base import BaseFeatureMetric

def compute_polynomial_mmd_averages(
    x : np.ndarray,
    y : np.ndarray,
    n_subsets : int = 50,
    subset_size : int = 1000,
    ret_var : bool = False,
    **kernel_args
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    m = min(x.shape[0], y.shape[0])
    mmds = np.zeros(n_subsets)
    if ret_var:
        vars = np.zeros(n_subsets)

    choice = np.random.choice

    for i in range(n_subsets):
        g = x[choice(len(x), subset_size, replace=False)]
        r = y[choice(len(y), subset_size, replace=False)]
        o = compute_polynomial_mmd(g, r, **kernel_args, var_at_m=m, ret_var=ret_var)
        if ret_var:
            mmds[i], vars[i] = o
        else:
            mmds[i] = o

    return (mmds, vars) if ret_var else mmds


def compute_polynomial_mmd(
    x : torch.Tensor,
    y : torch.Tensor,
    degree : int = 3,
    gamma : Optional[float] = None,
    coef0 : int = 1,
    var_at_m : Optional[int] = None,
    ret_var : bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Computes KID (polynomial MMD) for given sets of features, obtained from Inception net
    or any other feature extractor.
    Args:
        x: Samples from data distribution. Shape (N_samples, data_dim), dtype: torch.float32 in range 0 - 1.
        y: Samples from data distribution. Shape (N_samples, data_dim), dtype: torch.float32 in range 0 - 1
        degree: Degree of a polynomial functions used in kernels. Default: 3
        gamma: Kernel parameter. See paper for details
        coef0: Kernel parameter. See paper for details
        var_at_m: Kernel variance. Default is `None`
        ret_var: whether to return variance after the distance is computed.
                       This function will return Tuple[float, float] in this case.
    Returns:
        KID score and variance (optional).
    """
    # use  k(x, y) = (gamma <x, y> + coef0)^degree
    # default gamma is 1 / dim

    K_XX = polynomial_kernel(x.numpy(), degree=degree, gamma=gamma, coef0=coef0)
    K_YY = polynomial_kernel(y.numpy(), degree=degree, gamma=gamma, coef0=coef0)
    K_XY = polynomial_kernel(x.numpy(), y.numpy(), degree=degree, gamma=gamma, coef0=coef0)

    result = _mmd2_and_variance(K_XX, K_XY, K_YY, var_at_m=var_at_m, ret_var=ret_var)
    if not ret_var:
        return torch.from_numpy(np.array(result))
    else:
        return torch.from_numpy(np.array(result[0])), torch.from_numpy(np.array(result[1]))


def _mmd2_and_variance(
    K_XX : np.ndarray,
    K_XY : np.ndarray,
    K_YY : np.ndarray,
    unit_diagonal : bool = False,
    mmd_est : str = 'unbiased',
    var_at_m : Optional[int] = None,
    ret_var : bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    # based on
    # https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
    # but changed to not compute the full kernel matrix at once
    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)
    if var_at_m is None:
        var_at_m = m

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
        sum_diag2_X = sum_diag2_Y = m
    else:
        diag_X = np.diagonal(K_XX)
        diag_Y = np.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

        sum_diag2_X = _sqn(diag_X)
        sum_diag2_Y = _sqn(diag_Y)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)
    K_XY_sums_1 = K_XY.sum(axis=1)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if mmd_est == 'biased':
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2 * K_XY_sum / (m * m))
    else:
        assert mmd_est in {'unbiased', 'u-statistic'}
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m - 1))
        if mmd_est == 'unbiased':
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m - 1))

    if not ret_var:
        return mmd2

    Kt_XX_2_sum = _sqn(K_XX) - sum_diag2_X
    Kt_YY_2_sum = _sqn(K_YY) - sum_diag2_Y
    K_XY_2_sum = _sqn(K_XY)

    dot_XX_XY = Kt_XX_sums.dot(K_XY_sums_1)
    dot_YY_YX = Kt_YY_sums.dot(K_XY_sums_0)

    m1 = m - 1
    m2 = m - 2
    zeta1_est = (
        1 / (m * m1 * m2) *
        (_sqn(Kt_XX_sums) - Kt_XX_2_sum + _sqn(Kt_YY_sums) - Kt_YY_2_sum)
        - 1 / (m * m1) ** 2 * (Kt_XX_sum ** 2 + Kt_YY_sum ** 2)
        + 1 / (m * m * m1) * (
            _sqn(K_XY_sums_1) + _sqn(K_XY_sums_0) - 2 * K_XY_2_sum)
        - 2 / m ** 4 * K_XY_sum ** 2
        - 2 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 2 / (m ** 3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    zeta2_est = (
        1 / (m * m1) * (Kt_XX_2_sum + Kt_YY_2_sum)
        - 1 / (m * m1) ** 2 * (Kt_XX_sum ** 2 + Kt_YY_sum ** 2)
        + 2 / (m * m) * K_XY_2_sum
        - 2 / m ** 4 * K_XY_sum ** 2
        - 4 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 4 / (m ** 3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    var_est = (4 * (var_at_m - 2) / (var_at_m * (var_at_m - 1)) * zeta1_est
               + 2 / (var_at_m * (var_at_m - 1)) * zeta2_est)

    return mmd2, var_est


def _sqn(arr : np.ndarray) -> np.ndarray:
    flat = np.ravel(arr)
    return flat.dot(flat)

class KID(BaseFeatureMetric):
    r"""Creates a criterion that measures Kernel Inception Distance (polynomial MMD) for two datasets of images.

    Args:
        degree: Degree of a polynomial functions used in kernels. Default: 3
        gamma: Kernel parameter. See paper for details
        coef0: Kernel parameter. See paper for details
        var_at_m: Kernel variance. Default is `None`
        ret_var: Whether to return variance after the distance is computed.
                    This function will return Tuple[float, float] in this case. Default: False

    Reference:
        Demystifying MMD GANs https://arxiv.org/abs/1801.01401 
    """
    def __init__(
        self,
        degree : int = 3,
        gamma : Optional[float] = None,
        coef0 : int = 1,
        var_at_m : Optional[int] = None,
        ret_var : bool = False
        ) -> np.ndarray:
        super(KID, self).__init__()
        self.compute = partial(
            compute_polynomial_mmd,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            ret_var=ret_var)

    def forward(self, predicted_features: torch.Tensor, target_features: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Interface of Kernel Inception Distance.
        It's computed for a whole set of data and uses features from encoder instead of images itself to decrease
        computation cost. KID can compare two data distributions with different number of samples.
        But dimensionalities should match, otherwise it won't be possible to correctly compute statistics.

        Args:
            predicted_features: Low-dimension representation of predicted image set. Shape (N_pred, encoder_dim)
            target_features: Low-dimension representation of target image set. Shape (N_targ, encoder_dim)

        Returns:
            score: Scalar value of the distance between image sets features.
        """
        # Check inputs
        super(KID, self).forward(predicted_features, target_features)
        return self.compute(predicted_features, target_features)
