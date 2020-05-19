from typing import Optional, Union, Tuple, List

import torch


def _adjust_dimensions(x: torch.Tensor, y: torch.Tensor):
    r"""Expands input tensors dimensions to 4D
    """
    # TODO: add support of 5D tensors here or in the __compute_ssim function.
    num_dimentions = x.dim()
    if num_dimentions == 2:
        x = x.expand(1, 1, *x.shape)
        y = y.expand(1, 1, *y.shape)
    elif num_dimentions == 3:
        x = x.expand(1, *x.shape)
        y = y.expand(1, *y.shape)
    elif num_dimentions != 4:
        raise ValueError('Expected 2, 3, or 4 dimensions (got {})'.format(num_dimentions))

    return x, y


def _validate_input(
        x: torch.Tensor,
        y: torch.Tensor,
        kernel_size: Optional[int] = None,
        scale_weights: Union[Optional[Tuple[float]], Optional[List[float]], Optional[torch.Tensor]] = None) -> None:
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor), \
        f'Both images must be torch.Tensors, got {type(x)} and {type(y)}.'
    assert 1 < x.dim() < 5, f'Input images must be 2D, 3D or 4D tensors, got images of shape {x.size()}.'
    assert x.size() == y.size(), f'Input images must have the same dimensions, got {x.size()} and {y.size()}.'
    if kernel_size is not None:
        assert kernel_size % 2 == 1, f'Kernel size must be odd, got {kernel_size}.'
    if scale_weights is not None:
        assert isinstance(scale_weights, (list, tuple, torch.Tensor)), \
            f'Scale weights must be of type list, tuple or torch.Tensor, got {type(scale_weights)}.'
        assert (torch.Tensor(scale_weights).dim() == 1), \
            f'Scale weights must be one dimensional, got {torch.Tensor(scale_weights).dim()}.'
    return


def _validate_features(x: torch.Tensor, y: torch.Tensor, ) -> None:
    r"""Check, that computed features satisfy metric requirements.

    Args:
        x : Low-dimensional representation of predicted images.
        y : Low-dimensional representation of target images.
    """
    assert torch.is_tensor(x) and torch.is_tensor(y), \
        f"Both features should be torch.Tensors, got {type(x)} and {type(y)}"
    assert len(x.shape) == 2, \
        f"Predicted features must have shape (N_samples, encoder_dim), got {x.shape}"
    assert len(y.shape) == 2, \
        f"Target features must have shape  (N_samples, encoder_dim), got {y.shape}"
    assert x.shape[1] == y.shape[1], \
        f"Features dimensionalities should match, otherwise it won't be possible to correctly compute statistics. \
            Got {x.shape[1]} and {y.shape[1]}"
    assert x.device == y.device, "Both tensors should be on the same device"
