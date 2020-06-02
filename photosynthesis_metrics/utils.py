from typing import Optional, Union, Tuple, List

import torch


def _adjust_dimensions(input_tensors: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]):
    r"""Expands input tensors dimensions to 4D
    """
    if isinstance(input_tensors, torch.Tensor):
        input_tensors = (input_tensors,)

    for tensor in input_tensors:
        if tensor.dim() == 2:
            tensor.unsqueeze_(0)
        if tensor.dim() == 3:
            tensor.unsqueeze_(0)
        if tensor.dim() != 4 and tensor.dim() != 5:
            raise ValueError(f'Expected 2, 3, 4 or 5 dimensions (got {tensor.dim()})')

    if len(input_tensors) == 1:
        return input_tensors[0]
    return input_tensors


def _validate_input(
        input_tensors: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        kernel_size: Optional[int] = None,
        scale_weights: Union[Optional[Tuple[float]], Optional[List[float]], Optional[torch.Tensor]] = None) -> None:

    if isinstance(input_tensors, torch.Tensor):
        input_tensors = (input_tensors,)

    assert isinstance(input_tensors, tuple)
    assert 0 < len(input_tensors) < 3, f'Expected one or two input tensors, got {len(input_tensors)}'

    for tensor in input_tensors:
        assert isinstance(tensor, torch.Tensor), f'Both images must be torch.Tensors, got {type(tensor)}.'
        assert 1 < tensor.dim() < 6, f'Input images must be 2D, 3D or 4D tensors, got images of shape {tensor.size()}.'
        if tensor.dim() == 5:
            assert tensor.size(-1) == 2, f'Expected Complex 5D tensor with (N,C,H,W,2) size, got {tensor.size()}'

    if len(input_tensors) == 2:
        assert input_tensors[0].size() == input_tensors[1].size(), \
            f'Input images must have the same dimensions, got {input_tensors[0].size()} and {input_tensors[1].size()}.'

    if kernel_size is not None:
        assert kernel_size % 2 == 1, f'Kernel size must be odd, got {kernel_size}.'
    if scale_weights is not None:
        assert isinstance(scale_weights, (list, tuple, torch.Tensor)), \
            f'Scale weights must be of type list, tuple or torch.Tensor, got {type(scale_weights)}.'
        assert (torch.tensor(scale_weights).dim() == 1), \
            f'Scale weights must be one dimensional, got {torch.tensor(scale_weights).dim()}.'
    return


def _validate_features(x: torch.Tensor, y: torch.Tensor) -> None:
    r"""Check, that computed features satisfy metric requirements.

    Args:
        x : Low-dimensional representation of predicted images.
        y : Low-dimensional representation of target images.
    """
    assert torch.is_tensor(x) and torch.is_tensor(y), \
        f"Both features should be torch.Tensors, got {type(x)} and {type(y)}"
    assert x.dim() == 2, \
        f"Predicted features must have shape (N_samples, encoder_dim), got {x.shape}"
    assert y.dim() == 2, \
        f"Target features must have shape  (N_samples, encoder_dim), got {y.shape}"
    assert x.size(1) == y.size(1), \
        f"Features dimensionalities should match, otherwise it won't be possible to correctly compute statistics. \
            Got {x.size(1)} and {y.size(1)}"
    assert x.device == y.device, "Both tensors should be on the same device"
