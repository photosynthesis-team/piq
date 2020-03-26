import torch


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
