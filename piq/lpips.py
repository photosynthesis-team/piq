"""
Implementation of Learned Perceptual Image Patch Similarity (LPIPS) metric
References:
    Zhang, Richard and Isola, Phillip and Efros, et al. (2018)
    The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
    IEEE/CVF Conference on Computer Vision and Pattern Recognition
    https://arxiv.org/abs/1801.03924
    https://github.com/richzhang/PerceptualSimilarity
"""

from typing import List, Union, Collection

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torchvision.models import vgg16, vgg19

from piq.utils import _validate_input, _reduce
from piq.functional import similarity_map, L2Pool2d
from piq.perceptual import VGG16_LAYERS, VGG19_LAYERS, IMAGENET_MEAN, IMAGENET_STD


class LPIPS(_Loss):
    r"""Learned Perceptual Image Patch Similarity metric. Only VGG16 learned weights are supported.

    By default expects input to be in range [0, 1], which is then normalized by ImageNet statistics into range [-1, 1].
    If no normalisation is required, change `mean` and `std` values accordingly.

    Args:
        replace_pooling: Flag to replace MaxPooling layer with AveragePooling. See references for details.
        distance: Method to compute distance between features: ``'mse'`` | ``'mae'``.
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        mean: List of float values used for data standardization. Default: ImageNet mean.
            If there is no need to normalize data, use [0., 0., 0.].
        std: List of float values used for data standardization. Default: ImageNet std.
            If there is no need to normalize data, use [1., 1., 1.].

    Examples:
        >>> loss = LPIPS()
        >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> y = torch.rand(3, 3, 256, 256)
        >>> output = loss(x, y)
        >>> output.backward()

    References:
        Gatys, Leon and Ecker, Alexander and Bethge, Matthias (2016).
        A Neural Algorithm of Artistic Style
        Association for Research in Vision and Ophthalmology (ARVO)
        https://arxiv.org/abs/1508.06576

        Zhang, Richard and Isola, Phillip and Efros, et al. (2018)
        The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
        IEEE/CVF Conference on Computer Vision and Pattern Recognition
        https://arxiv.org/abs/1801.03924
        https://github.com/richzhang/PerceptualSimilarity
    """
    _weights_url = "https://github.com/photosynthesis-team/" + \
        "photosynthesis.metrics/releases/download/v0.4.0/lpips_weights.pt"

    def __init__(self, replace_pooling: bool = False, distance: str = "mse", reduction: str = "mean",
                 mean: List[float] = IMAGENET_MEAN, std: List[float] = IMAGENET_STD,) -> None:
        super().__init__()

        lpips_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']
        lpips_weights = torch.hub.load_state_dict_from_url(self._weights_url, progress=False)
        
        super().__init__("vgg16", layers=lpips_layers, weights=lpips_weights,
                         replace_pooling=replace_pooling, distance=distance,
                         reduction=reduction, mean=mean, std=std,
                         normalize_features=True)

        if callable(feature_extractor):
            self.model = feature_extractor
            self.layers = layers
        elif feature_extractor == "vgg16":
            self.model = vgg16(pretrained=True, progress=False).features
            self.layers = [VGG16_LAYERS[l] for l in layers]
        elif feature_extractor == "vgg19":
            self.model = vgg19(pretrained=True, progress=False).features
            self.layers = [VGG19_LAYERS[l] for l in layers]
        else:
            raise ValueError("Unknown feature extractor")

        if replace_pooling:
            self.model = self.replace_pooling(self.model)

        # Disable gradients
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.distance = {
            "mse": nn.MSELoss,
            "mae": nn.L1Loss,
        }[distance](reduction='none')

        self.weights = [torch.tensor(w) if not isinstance(w, torch.Tensor) else w for w in weights]

        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)
        
        self.normalize_features = normalize_features
        self.reduction = reduction



