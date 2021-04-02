from typing import Tuple, List, Optional
import torch


def _validate_input(
    tensors: List[torch.Tensor],
    dim_range: Tuple[int, int] = (0, -1),
    data_range: Tuple[float, float] = (0., -1.),
    # size_dim_range: Tuple[float, float] = (0., -1.),
    size_range: Optional[Tuple[int, int]] = None,
) -> None:
    r"""Check that input(-s)  satisfies the requirements
    Args:
        tensors: Tensors to check
        dim_range: Allowed number of dimensions. (min, max)
        data_range: Allowed range of values in tensors. (min, max)
        size_range: Dimensions to include in size comparison. (start_dim, end_dim + 1)
    """

    if not __debug__:
        return

    x = tensors[0]

    for t in tensors:
        assert torch.is_tensor(t), f'Expected torch.Tensor, got {type(t)}'
        assert t.device == x.device, f'Expected tensors to be on {x.device}, got {t.device}'

        if size_range is None:
            assert t.size() == x.size(), f'Expected tensors with same size, got {t.size()} and {x.size()}'
        else:
            assert t.size()[size_range[0]: size_range[1]] == x.size()[size_range[0]: size_range[1]], \
                f'Expected tensors with same size at given dimensions, got {t.size()} and {x.size()}'

        if dim_range[0] == dim_range[1]:
            assert t.dim() == dim_range[0], f'Expected number of dimensions to be {dim_range[0]}, got {t.dim()}'
        elif dim_range[0] < dim_range[1]:
            assert dim_range[0] <= t.dim() <= dim_range[1], \
                f'Expected number of dimensions to be between {dim_range[0]} and {dim_range[1]}, got {t.dim()}'

        if data_range[0] < data_range[1]:
            assert data_range[0] <= t.min(), \
                f'Expected values to be greater or equal to {data_range[0]}, got {t.min()}'
            assert t.max() <= data_range[1], \
                f'Expected values to be lower or equal to {data_range[1]}, got {t.max()}'


def _reduce(x: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    r"""Reduce input in batch dimension if needed.

    Args:
        x: Tensor with shape (N, *).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``
    """
    if reduction == 'none':
        return x
    elif reduction == 'mean':
        return x.mean(dim=0)
    elif reduction == 'sum':
        return x.sum(dim=0)
    else:
        raise ValueError("Uknown reduction. Expected one of {'none', 'mean', 'sum'}")


def _version_tuple(v):
    # Split by dot and plus
    return tuple(map(int, v.split('+')[0].split('.')))
