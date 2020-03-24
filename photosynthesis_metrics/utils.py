from typing import Callable

import torch
import numpy as np

from photosynthesis_metrics.feature_extractors.fid_inception import InceptionV3

class BaseFeatureMetric(torch.nn.Module):
    r"""Base class for all metrics, which require computation of per image features.
     For example: FID, KID, MSID etc.
     """

    def __init__(self) -> None:
        super(BaseFeatureMetric, self).__init__()

    def forward(self, predicted_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        # Sanity check for input
        assert predicted_features.shape[1] == target_features.shape[1], \
            f"Features dimensionalities should match, otherwise it won't be possible to correctly compute statistics. \
                Got {predicted_features.shape[1]} and {target_features.shape[1]}"
        assert len(predicted_features.shape) == 2, \
            f"Predicted features must have shape  (N_samples, encoder_dim), got {predicted_features.shape}"
        assert len(target_features.shape) == 2, \
            f"Target features must have shape  (N_samples, encoder_dim), got {target_features.shape}"

    def _compute_feats(
        self,
        loader: torch.utils.data.DataLoader,
        feature_extractor : Callable = None,
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

        feature_extractor.eval()

        total_feats = []
        for batch in loader:
            images = batch['images']
            N = images.shape[0]
            images = images.float().to(device)

            # Getting features
            batch_feats = feature_extractor(images).view(N, -1)
            total_feats.append(batch_feats)
        return torch.Tensor(total_feats)


