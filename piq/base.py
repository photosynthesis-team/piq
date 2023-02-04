import torch

from piq.feature_extractors import InceptionV3


class BaseFeatureMetric(torch.nn.Module):
    r"""Base class for all metrics, which require computation of per image features.
     For example: FID, KID, MSID etc.
     """

    def __init__(self) -> None:
        super(BaseFeatureMetric, self).__init__()

    def forward(self, x_features: torch.Tensor, y_features: torch.Tensor) -> torch.Tensor:
        return self.compute_metric(x_features, y_features)

    @torch.no_grad()
    def compute_feats(
            self,
            loader: torch.utils.data.DataLoader,
            feature_extractor: torch.nn.Module = None,
            device: str = 'cuda') -> torch.Tensor:
        r"""Generate low-dimensional image descriptors

        Args:
            loader: Should return dict with key `images` in it
            feature_extractor: model used to generate image features, if None use `InceptionNetV3` model.
                Model should return a list with features from one of the network layers.
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
            # TODO(jamil 26.03.20): Add support for more than one feature map
            assert len(features) == 1, \
                f"feature_encoder must return list with features from one layer. Got {len(features)}"

            features = features[0].view(N, -1)
            features = features.cpu()
            total_feats.append(features)
            torch.cuda.empty_cache()

        feature_extractor.cpu()
        torch.cuda.empty_cache()
        return torch.cat(total_feats, dim=0)

    def compute_metric(self, x_features: torch.Tensor, y_features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This function should be defined for each children class")
