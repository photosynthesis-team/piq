"""
Implementation of VGG16 loss, originaly used for style transfer and usefull in many other task (including GAN training)
It's work in progress, no guarantees that code will work
"""
# import collections
from typing import List, Union

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torchvision.models import vgg16, vgg19

from photosynthesis_metrics.utils import _validate_input, _adjust_dimensions


# Map VGG names to corresponding number in torchvision layer
VGG16_LAYERS = {
    "conv1_1": 0, "relu1_1": 1,
    "conv1_2": 2, "relu1_2": 3,
    "pool1": 4,
    "conv2_1": 5, "relu2_1": 6,
    "conv2_2": 7, "relu2_2": 8,
    "pool2": 9,
    "conv3_1": 10, "relu3_1": 11,
    "conv3_2": 12, "relu3_2": 13,
    "conv3_3": 14, "relu3_3": 15,
    "pool3": 16,
    "conv4_1": 17, "relu4_1": 18,
    "conv4_2": 19, "relu4_2": 20,
    "conv4_3": 21, "relu4_3": 22,
    "pool4": 23,
    "conv5_1": 24, "relu5_1": 25,
    "conv5_2": 26, "relu5_2": 27,
    "conv5_3": 28, "relu5_3": 29,
    "pool5": 30,
}

VGG19_LAYERS = {
    "conv1_1": 0, "relu1_1": 1,
    "conv1_2": 2, "relu1_2": 3,
    "pool1": 4,
    "conv2_1": 5, "relu2_1": 6,
    "conv2_2": 7, "relu2_2": 8,
    "pool2": 9,
    "conv3_1": 10, "relu3_1": 11,
    "conv3_2": 12, "relu3_2": 13,
    "conv3_3": 14, "relu3_3": 15,
    "conv3_4": 16, "relu3_4": 17,
    "pool3": 18,
    "conv4_1": 19, "relu4_1": 20,
    "conv4_2": 21, "relu4_2": 22,
    "conv4_3": 23, "relu4_3": 24,
    "conv4_4": 25, "relu4_4": 26,
    "pool4": 27,
    "conv5_1": 28, "relu5_1": 29,
    "conv5_2": 30, "relu5_2": 31,
    "conv5_3": 32, "relu5_3": 33,
    "conv5_4": 34, "relu5_4": 35,
    "pool5": 36,
}


class ContentLoss(_Loss):
    r"""Creates Content loss that can be used for image style transfer of as a measure in
    image to image tasks.
    Uses pretrained VGG models from torchvision. Normalizes features before summation.
    Expects input to be in range [0, 1] or normalized with ImageNet statistics into range [-1, 1]
    """
    # Constant used in feature normalization to avoid zero division
    EPS = 1e-10

    def __init__(self, feature_extractor: str = "vgg16", layers: List[str] = ["relu3_3"],
                 weights: List[Union[float, torch.Tensor]] = [1.], replace_pooling: bool = False, distance: str = "mse",
                 reduction: str = "mean", normalize_input: bool = True, normalize_features: bool = False) -> None:
        r"""
        Args:
            feature_extractor: Name of model used to extract features. One of {`vgg16`, `vgg19`}
            layers: List of string with layer names. Default: [`relu3_3`]
            weights: List of float weight to balance different layers
            replace_pooling: Flag to replace MaxPooling layer with AveragePooling. See [1] for details.
            distance: Method to compute distance between features. One of {`mse`, `mae`}.
            reduction: Reduction over samples in batch: "mean"|"sum"|"none"
            normalize_input: If true, scales the input from range (0, 1) to the range the
                pretrained VGG network expects, namely (-1, 1)
            normalize_features: If true, unit-normalize each feature in channel dimension before scaling
                and computing distance. See [2] for details.

        References:
            .. [1] Gatys, Leon and Ecker, Alexander and Bethge, Matthias
            (2016). A Neural Algorithm of Artistic Style}
            Association for Research in Vision and Ophthalmology (ARVO)
            https://arxiv.org/abs/1508.06576
    
            .. [2] Zhang, Richard and Isola, Phillip and Efros, et al.
            (2018) The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
            2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition
            https://arxiv.org/abs/1801.03924

        """
        super().__init__()

        if feature_extractor == "vgg16":
            self.model = vgg16(pretrained=True, progress=False)
            self.layers = [VGG16_LAYERS[l] for l in layers]
        elif feature_extractor == "vgg19":
            self.model = vgg19(pretrained=True, progress=False)
            self.layers = [VGG19_LAYERS[l] for l in layers]
        else:
            raise ValueError("Unknown feature extractor")

        if replace_pooling:
            for i, layer in enumerate(self.model.features):
                if isinstance(layer, torch.nn.MaxPool2d):
                    self.model.features[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # Disable gradients
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.distance = {
            "mse": nn.MSELoss(reduction='none'),
            "mae": nn.L1Loss(reduction='none'),
        }[distance]

        self.weights = weights
        mean = torch.tensor([0.485, 0.456, 0.406]) if normalize_input else torch.tensor([0., 0., 0.])
        std = torch.tensor([0.229, 0.224, 0.225]) if normalize_input else torch.tensor([1., 1., 1.])
        self.mean = mean.view(1, 3, 1, 1)
        self.std = std.view(1, 3, 1, 1)
        
        self.normalize_features = normalize_features

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Computation of Content loss between feature representations of prediction and target tensors.

        Args:
            prediction: Tensor of prediction of the network.
            target: Reference tensor.

        """
        _validate_input(input_tensors=(prediction, target), allow_5d=False)
        prediction, target = _adjust_dimensions(input_tensors=(prediction, target))
        
        # Normalize input
        prediction = (prediction - self.mean) / self.std
        target = (target - self.mean) / self.std

        prediction_features = self.get_features(prediction)
        target_features = self.get_features(target)
        return self.compute_metric(prediction_features, target_features)
    
    def compute_metric(self, prediction_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        distances = [self.distance(x, y) for x, y in zip(prediction_features, target_features)]

        # Scale distances, then average in spatial dimensions and sum in channel dimensions
        loss = torch.cat([(d * w).mean(dim=[2, 3]) for d, w in zip(distances, self.weights)]).sum(dim=1)

        # Solve big memory consumption
        torch.cuda.empty_cache()

        if self.reduction == 'none':
            return loss

        return {'mean': loss.mean,
                'sum': loss.sum
                }[self.reduction](dim=0)

    def get_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: torch.Tensor with shape (N, C, H, W)
        
        Returns:
            features: List of features extracted from intermediate layers
        """
        features = []
        for name, module in self.model.features._modules.items():
            x = module(x)
            if int(name) in self.layers:
                features.append(self.normalize(x) if self.normalize_features else x)
        return features

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize feature maps in channel direction to unit length.
        Args:
            x: Tensor with shape (N, C, H, W)
        Returns:
            x_norm: Normalized input
        """
        norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
        return x / (norm_factor + self.EPS)


class StyleLoss(ContentLoss):
    r"""Creates Style loss that can be used for image style transfer or as a measure in
    image to image tasks.
    Uses pretrained VGG models from torchvision. Features can be normalized before summation.
    Expects input to be in range [0, 1] or normalized with ImageNet statistics into range [-1, 1]
    """

    def compute_metric(self, prediction_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        prediction_gram = [self.gram_matrix(x) for x in prediction_features]
        target_gram = [self.gram_matrix(x) for x in target_features]

        distances = [self.distance(x, y) for x, y in zip(prediction_gram, target_gram)]

        # Scale distances, then average in spatial dimensions and sum in channel dimensions
        loss = torch.stack([(d * w).mean(dim=[2, 3]) for d, w in zip(distances, self.weights)]).sum(dim=1)

        # Solve big memory consumption
        torch.cuda.empty_cache()

        if self.reduction == 'none':
            return loss

        return {'mean': loss.mean,
                'sum': loss.sum
                }[self.reduction](dim=0)

    def gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute Gram matrix for batch of features.
        Args:
            x: Tensor of shape BxCxHxW
        """

        B, C, H, W = x.size()
        gram = []
        for i in range(B):
            x = x[i].view(C, H * W)
            gram.append(torch.mm(x, x.t()))
        return gram


class LPIPS(ContentLoss):
    r"""Learned Perceptual Image Patch Similarity metric.
    For now only VGG16 learned weights are supported.
    Expects input to be in range [0, 1] or normalized with ImageNet statistics into range [-1, 1]
    """
    _weights_url = "https://github.com/photosynthesis-team/" + \
        "photosynthesis.metrics/releases/download/v0.4.0/lpips_weights.pt"

    def __init__(self, feature_extractor: str = "vgg16",
                 replace_pooling: bool = False, distance: str = "mse",
                 reduction: str = "mean", normalize_input: bool = True) -> None:
        r"""
        Args:
            feature_extractor: Name of model used to extract features. One of {`vgg16`, `vgg19`}
            replace_pooling: Flag to replace MaxPooling layer with AveragePooling. See [1] for details.
            distance: Method to compute distance between features. One of {`mse`, `mae`}.
            reduction: Reduction over samples in batch: "mean"|"sum"|"none"
            normalize_input: If true, scales the input from range (0, 1) to the range the
                pretrained VGG network expects, namely (-1, 1)

        References:
            .. [1] Gatys, Leon and Ecker, Alexander and Bethge, Matthias
            (2016). A Neural Algorithm of Artistic Style}
            Association for Research in Vision and Ophthalmology (ARVO)
            https://arxiv.org/abs/1508.06576
    
            .. [2] Zhang, Richard and Isola, Phillip and Efros, et al.
            (2018) The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
            2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition
            https://arxiv.org/abs/1801.03924

        """
        lpips_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_4']
        lpips_weights = torch.hub.load_state_dict_from_url(self._weights_url)
        super().__init__("vgg16", layers=lpips_layers, weights=lpips_weights,
                         replace_pooling=replace_pooling, distance=distance,
                         reduction=reduction, normalize_input=normalize_input,
                         normalize_features=True)
