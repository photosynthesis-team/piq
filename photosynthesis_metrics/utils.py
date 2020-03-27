from typing import Optional, Union, Tuple, List


import torch


def _adjust_dimensions(x: torch.Tensor, y: torch.Tensor):
    r"""Expands input tensors dimensions to 4D
    """
    # TODO: try to move this block in __compute_ssim since it is very general.
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


def _validate_input(x: torch.Tensor, y: torch.Tensor, kernel_size: Optional[int] = None,
                    scale_weights: Union[Optional[Tuple[float]], Optional[List[float]]] = None) -> None:
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor),\
        f'Both images must be torch.Tensors, got {type(x)} and {type(y)}.'
    assert len(x.shape) == 4, f'Input images must be 4D tensors, got images of shape {x.shape}.'
    assert x.shape == y.shape, f'Input images must have the same dimensions, got {x.shape} and {y.shape}.'
    if kernel_size is not None:
        assert kernel_size % 2 == 1, f'Kernel size must be odd, got {kernel_size}.'
    if scale_weights is not None:
        assert isinstance(scale_weights, (list, tuple)), \
            f'Scale weights must be of type list or tuple, got {type(scale_weights)}.'
        assert len(scale_weights) == 4, f'Scale weights collection must contain 4 values, got {len(scale_weights)}.'
    
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