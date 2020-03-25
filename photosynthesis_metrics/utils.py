from typing import Callable, Optional, Union, Tuple, List

import torch
import numpy as np


from photosynthesis_metrics.feature_extractors.fid_inception import InceptionV3

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

class BaseFeatureMetric(torch.nn.Module):
    r"""Base class for all metrics, which require computation of per image features.
     For example: FID, KID, MSID etc.
     """

    def __init__(self) -> None:
        super(BaseFeatureMetric, self).__init__()

    def forward(self, predicted_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        # Sanity check for input
        _validate_features(predicted_features, target_features)
        
    def _compute_feats(
        self,
        loader: torch.utils.data.DataLoader,
        feature_extractor : torch.nn.Module = None,
        device : str = 'cuda') -> torch.Tensor:
        r"""Generate low-dimensional image desciptors to be used for computing MSID score.
        Args:
            loader: Should return dict with key `images` in it
            feature_extractor: model used to generate image features, if None use `InceptionNetV3` model.
                Model should return a 
            out_features: size of `feature_extractor` output
            device: Device on which to compute inference of the model
        """

        if feature_extractor is None:
            print('WARNING: default feature extractor (InceptionNet V2) is used.')
            feature_extractor = InceptionV3()
        else:
            assert isinstance(feature_extractor, torch.nn.Module), \
                f"Feature extractor must be PyTorch module. Got {type(feature_extractor)}"
        feature_extractor.to(device)
        feature_extractor.eval()

        total_feats = []
        for batch in loader:
            images = batch['images']
            N = images.shape[0]
            images = images.float().to(device)

            # Get features
            features = feature_extractor(images)
            assert len(features) == 1, \
                f"feature_encoder must return list with features from one layer. Got {len(features)}"
            total_feats.append(features[0].view(N, -1))

        return torch.cat(total_feats, dim=0)
