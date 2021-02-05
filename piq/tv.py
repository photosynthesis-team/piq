r"""Implemetation of Total Variation metric, based on article
 remi.flamary.com/demos/proxtv.html and www.wikiwand.com/en/Total_variation_denoising
"""

import torch
from torch.nn.modules.loss import _Loss
from piq.utils import _validate_input, _adjust_dimensions


def total_variation(x: torch.Tensor, reduction: str = 'mean', norm_type: str = 'l2') -> torch.Tensor:
    r"""Compute Total Variation metric
    Args:
        x: Tensor with shape (N, C, H, W).
        reduction: Reduction over samples in batch: "mean"|"sum"|"none"
        norm_type: {'l1', 'l2', 'l2_squared'}, defines which type of norm to implement, isotropic  or anisotropic.

    Returns:
        score : Total variation of a given tensor
    Description:
        ## 1D signal series 
        For a digital signal $y_{n}$, we can, for example, define the total variation as
        $$
        V(y)=\sum _{n}|y_{n+1}-y_{n}|.
        $$
        Given an input signal $x_{n}$, the goal of total variation denoising is to find an approximation, call it $y_{n}$, that has smaller total variation than $x_{n}$ but is "close" to $x_{n}$. One measure of closeness is the sum of square errors:
        $$
        \mathrm{E}(x, y)=\frac{1}{n} \sum_{n}\left(x_{n}-y_{n}\right)^{2}.
        $$

        So the total-variation denoising problem amounts to minimizing the following discrete functional over the signal $y_{n}$:
        $$
        \mathrm {E} (x,y)+\lambda V(y).
        $$

        By differentiating this functional with respect to $y_{n}$, we can derive a corresponding Euler–Lagrange equation, that can be numerically integrated with the original signal $x_{n}$ as initial condition. This was the original approach. Alternatively, since this is a convex functional, techniques from convex optimization can be used to minimize it and find the solution $y_{n}$.

        ## Regularization properties

        The regularization parameter $\lambda$ plays a critical role in the denoising process. When $\lambda =0$, there is no smoothing and the result is the same as minimizing the sum of squares. As $\lambda \to \infty $, however, the total variation term plays an increasingly strong role, which forces the result to have smaller total variation, at the expense of being less like the input (noisy) signal. Thus, the choice of regularization parameter is critical to achieving just the right amount of noise removal.
    
        ## 2D signal images

        We now consider 2D signals y, such as images. The total-variation norm proposed by the 1992 article is
        $$
        V(y)=\sum_{i, j} \sqrt{\left|y_{i+1, j}-y_{i, j}\right|^{2}}+\sqrt{\left|y_{i, j+1}-y_{i, j}\right|^{2}}
        $$
        and is isotropic and not differentiable. A variation that is sometimes used, since it may sometimes be easier to minimize, is an anisotropic version
        $$
        V_{\text {aniso }}(y)=\sum_{i, j} \sqrt{\left|y_{i+1, j}-y_{i, j}\right|^{2}}+\sqrt{\left|y_{i, j+1}-y_{i, j}\right|^{2}}=\sum_{i, j}\left|y_{i+1, j}-y_{i, j}\right|+\left|y_{i, j+1}-y_{i, j}\right|
        $$

        The standard total-variation denoising problem is still of the form
        $$
            {\displaystyle \min _{y}[\operatorname {E} (x,y)+\lambda V(y)],}
        $$
        where $E$ is the 2D $L_2$ norm. In contrast to the 1D case, solving this denoising is non-trivial. A recent algorithm that solves this is known as the primal dual method.

        Due in part to much research in compressed sensing in the mid-2000s, there are many algorithms, such as the split-Bregman method, that solve variants of this problem.

        ## Rudin–Osher–Fatemi PDE

        Suppose that we are given a noisy image $f$ and wish to compute a denoised image $u$ over a 2D space. ROF showed that the minimization problem we are looking to solve is:
        $$
            {\displaystyle \min _{u\in \operatorname {BV} (\Omega )}\;\|u\|_{\operatorname {TV} (\Omega )}+{\lambda \over 2}\int _{\Omega }(f-u)^{2}\,dx}
        $$
        where ${\textstyle \operatorname {BV} (\Omega )}$ is the set of functions with bounded variation over the domain $\Omega$ , ${\textstyle \operatorname {TV} (\Omega )}$ is the total variation over the domain, and ${\textstyle \lambda }$ is a penalty term. When ${\textstyle u}$ is smooth, the total variation is equivalent to the integral of the gradient magnitude:
        $$
            {\displaystyle \|u\|_{\operatorname {TV} (\Omega )}=\int _{\Omega }\|\nabla u\|\,dx}
        $$
        where ${\textstyle \|\cdot \|}$ is the Euclidean norm. Then the objective function of the minimization problem becomes:
        $$
        {\displaystyle \min _{u\in \operatorname {BV} (\Omega )}\;\int _{\Omega }\left[\|\nabla u\|+{\lambda \over 2}(f-u)^{2}\right]\,dx}
        $$
        From this functional, the Euler-Lagrange equation for minimization – assuming no time-dependence – gives us the nonlinear elliptic partial differential equation:
        $$
        \left\{\begin{array}{ll}
        \nabla \cdot\left(\frac{\nabla u}{\|\nabla u\|}\right)+\lambda(f-u)=0, & u \in \Omega \\
        \frac{\partial u}{\partial n}=0, & u \in \partial \Omega
        \end{array}\right.
        $$

        For some numerical algorithms, it is preferable to instead solve the time-dependent version of the ROF equation:
        $$
        \frac{\partial u}{\partial t}=\nabla \cdot\left(\frac{\nabla u}{\|\nabla u\|}\right)+\lambda(f-u)
        $$

        ## Applications

        The Rudin–Osher–Fatemi model was a pivotal component in producing the first image of a black hole.

    References:
        https://www.wikiwand.com/en/Total_variation_denoising
        https://remi.flamary.com/demos/proxtv.html
    """
    _validate_input(x, allow_5d=False)
    x = _adjust_dimensions(x)

    if norm_type == 'l1':
        w_variance = torch.sum(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]), dim=[1, 2, 3])
        h_variance = torch.sum(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]), dim=[1, 2, 3])
        score = (h_variance + w_variance)
    elif norm_type == 'l2':
        w_variance = torch.sum(torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2), dim=[1, 2, 3])
        h_variance = torch.sum(torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2), dim=[1, 2, 3])
        score = torch.sqrt(h_variance + w_variance)
    elif norm_type == 'l2_squared':
        w_variance = torch.sum(torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2), dim=[1, 2, 3])
        h_variance = torch.sum(torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2), dim=[1, 2, 3])
        score = (h_variance + w_variance)
    else:
        raise ValueError("Incorrect reduction type, should be one of {'l1', 'l2', 'l2_squared'}")

    if reduction == 'none':
        return score

    return {'mean': score.mean,
            'sum': score.sum
            }[reduction](dim=0)


class TVLoss(_Loss):
    r"""Creates a criterion that measures the total variation of the
    the given input :math:`x`.


    If :attr:`norm_type` set to ``'l2'`` the loss can be described as:

    .. math::
        TV(x) = \sum_{N}\sqrt{\sum_{H, W, C}(|x_{:, :, i+1, j} - x_{:, :, i, j}|^2 +
        |x_{:, :, i, j+1} - x_{:, :, i, j}|^2)}

    Else if :attr:`norm_type` set to ``'l1'``:

    .. math::
        TV(x) = \sum_{N}\sum_{H, W, C}(|x_{:, :, i+1, j} - x_{:, :, i, j}| +
        |x_{:, :, i, j+1} - x_{:, :, i, j}|)

    where :math:`N` is the batch size, `C` is the channel size.

    Args:
        norm_type: one of {'l1', 'l2', 'l2_squared'}
        reduction: Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
    Shape:
        - Input: Required to be 2D (H, W), 3D (C,H,W) or 4D (N,C,H,W)

    Examples::

        >>> loss = TVLoss()
        >>> prediction = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> output = loss(prediction)
        >>> output.backward()

    References:
        https://www.wikiwand.com/en/Total_variation_denoising
        https://remi.flamary.com/demos/proxtv.html
    """

    def __init__(self, norm_type: str = 'l2', reduction: str = 'mean'):
        super().__init__()

        self.norm_type = norm_type
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor) -> torch.Tensor:
        r"""Computation of Total Variation (TV) index as a loss function.

        Args:
            prediction: Tensor of prediction of the network.

        Returns:
            Value of TV loss to be minimized.
        """
        score = total_variation(
            prediction,
            reduction=self.reduction,
            norm_type=self.norm_type,
        )
        return score
