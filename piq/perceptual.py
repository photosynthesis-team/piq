"""
Implementation of Content loss, Style loss and LPIPS metric
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
from typing import List, Union, Callable, Tuple

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torchvision.models import vgg16, vgg19

from piq.utils import _validate_input, _adjust_dimensions
from piq.functional import similarity_map, L2Pool2d


# Map VGG names to corresponding number in torchvision layer
VGG16_LAYERS = {
    "conv1_1": '0', "relu1_1": '1',
    "conv1_2": '2', "relu1_2": '3',
    "pool1": '4',
    "conv2_1": '5', "relu2_1": '6',
    "conv2_2": '7', "relu2_2": '8',
    "pool2": '9',
    "conv3_1": '10', "relu3_1": '11',
    "conv3_2": '12', "relu3_2": '13',
    "conv3_3": '14', "relu3_3": '15',
    "pool3": '16',
    "conv4_1": '17', "relu4_1": '18',
    "conv4_2": '19', "relu4_2": '20',
    "conv4_3": '21', "relu4_3": '22',
    "pool4": '23',
    "conv5_1": '24', "relu5_1": '25',
    "conv5_2": '26', "relu5_2": '27',
    "conv5_3": '28', "relu5_3": '29',
    "pool5": '30',
}

VGG19_LAYERS = {
    "conv1_1": '0', "relu1_1": '1',
    "conv1_2": '2', "relu1_2": '3',
    "pool1": '4',
    "conv2_1": '5', "relu2_1": '6',
    "conv2_2": '7', "relu2_2": '8',
    "pool2": '9',
    "conv3_1": '10', "relu3_1": '11',
    "conv3_2": '12', "relu3_2": '13',
    "conv3_3": '14', "relu3_3": '15',
    "conv3_4": '16', "relu3_4": '17',
    "pool3": '18',
    "conv4_1": '19', "relu4_1": '20',
    "conv4_2": '21', "relu4_2": '22',
    "conv4_3": '23', "relu4_3": '24',
    "conv4_4": '25', "relu4_4": '26',
    "pool4": '27',
    "conv5_1": '28', "relu5_1": '29',
    "conv5_2": '30', "relu5_2": '31',
    "conv5_3": '32', "relu5_3": '33',
    "conv5_4": '34', "relu5_4": '35',
    "pool5": '36',
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Constant used in feature normalization to avoid zero division
EPS = 1e-10


class ContentLoss(_Loss):
    r"""Creates Content loss that can be used for image style transfer of as a measure in
    image to image tasks.
    Uses pretrained VGG models from torchvision. Normalizes features before summation.
    Expects input to be in range [0, 1] or normalized with ImageNet statistics into range [-1, 1]
    Args:
        feature_extractor: Model to extract features or model name in {`vgg16`, `vgg19`}.
        layers: List of strings with layer names. Default: [`relu3_3`]
        weights: List of float weight to balance different layers
        replace_pooling: Flag to replace MaxPooling layer with AveragePooling. See [1] for details.
        distance: Method to compute distance between features. One of {`mse`, `mae`}.
        reduction: Reduction over samples in batch: "mean"|"sum"|"none"
        mean: List of float values used for data standartization. Default: ImageNet mean.
            If there is no need to normalize data, use [0., 0., 0.].
        std: List of float values used for data standartization. Default: ImageNet std.
            If there is no need to normalize data, use [1., 1., 1.].
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

    def __init__(self, feature_extractor: Union[str, Callable] = "vgg16", layers: Tuple[str] = ("relu3_3", ),
                 weights: List[Union[float, torch.Tensor]] = [1.], replace_pooling: bool = False,
                 distance: str = "mse", reduction: str = "mean", mean: List[float] = IMAGENET_MEAN,
                 std: List[float] = IMAGENET_STD, normalize_features: bool = False) -> None:

        super().__init__()

        if callable(feature_extractor):
            self.model = feature_extractor
            self.layers = layers
        else:
            if feature_extractor == "vgg16":
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

        self.weights = [torch.tensor(w) for w in weights]
        mean = torch.tensor(mean)
        std = torch.tensor(std)
        self.mean = mean.view(1, -1, 1, 1)
        self.std = std.view(1, -1, 1, 1)
        
        self.normalize_features = normalize_features
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Computation of Content loss between feature representations of prediction and target tensors.
        Args:
            prediction: Tensor of prediction of the network.
            target: Reference tensor.
        """
        _validate_input(input_tensors=(prediction, target), allow_5d=False)
        prediction, target = _adjust_dimensions(input_tensors=(prediction, target))

        self.model.to(prediction)
        prediction_features = self.get_features(prediction)
        target_features = self.get_features(target)

        distances = self.compute_distance(prediction_features, target_features)

        # Scale distances, then average in spatial dimensions, then stack and sum in channels dimension
        loss = torch.cat([(d * w.to(d)).mean(dim=[2, 3]) for d, w in zip(distances, self.weights)], dim=1).sum(dim=1)

        if self.reduction == 'none':
            return loss

        return {'mean': loss.mean,
                'sum': loss.sum
                }[self.reduction](dim=0)

    def compute_distance(self, prediction_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        r"""Take L2 or L1 distance between feature maps"""
        return [self.distance(x, y) for x, y in zip(prediction_features, target_features)]

    def get_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        r"""
        Args:
            x: torch.Tensor with shape (N, C, H, W)
        
        Returns:
            features: List of features extracted from intermediate layers
        """
        # Normalize input
        x = (x - self.mean.to(x)) / self.std.to(x)

        features = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.layers:
                features.append(self.normalize(x) if self.normalize_features else x)
        return features

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        r"""Normalize feature maps in channel direction to unit length.
        Args:
            x: Tensor with shape (N, C, H, W)
        Returns:
            x_norm: Normalized input
        """
        norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
        return x / (norm_factor + EPS)

    def replace_pooling(self, module: torch.nn.Module) -> torch.nn.Module:
        r"""Turn All MaxPool layers into AveragePool"""
        module_output = module
        if isinstance(module, torch.nn.MaxPool2d):
            module_output = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            
        for name, child in module.named_children():
            module_output.add_module(name, self.replace_pooling(child))
        return module_output


class StyleLoss(ContentLoss):
    r"""Creates Style loss that can be used for image style transfer or as a measure in
    image to image tasks. Computes distance between Gram matrixes of feature maps.
    Uses pretrained VGG models from torchvision. Features can be normalized before summation.
    Expects input to be in range [0, 1] or normalized with ImageNet statistics into range [-1, 1]
    """

    def compute_distance(self, prediction_features: torch.Tensor, target_features: torch.Tensor):
        """Take L2 or L1 distance between Gram matrixes of feature maps"""
        prediction_gram = [self.gram_matrix(x) for x in prediction_features]
        target_gram = [self.gram_matrix(x) for x in target_features]
        return [self.distance(x, y) for x, y in zip(prediction_gram, target_gram)]

    def gram_matrix(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute Gram matrix for batch of features.
        Args:
            x: Tensor of shape BxCxHxW
        """
        B, C, H, W = x.size()
        gram = []
        for i in range(B):
            features = x[i].view(C, H * W)

            # Add fake channel dimension
            gram.append(torch.mm(features, features.t()).unsqueeze(0))
        return torch.stack(gram)


class LPIPS(ContentLoss):
    r"""Learned Perceptual Image Patch Similarity metric.
    For now only VGG16 learned weights are supported.
    Expects input to be in range [0, 1] or normalized with ImageNet statistics into range [-1, 1]
    Args:
        feature_extractor: Name of model used to extract features. One of {`vgg16`, `vgg19`}
        use_average_pooling: Flag to replace MaxPooling layer with AveragePooling. See [1] for details.
        distance: Method to compute distance between features. One of {`mse`, `mae`}.
        reduction: Reduction over samples in batch: "mean"|"sum"|"none"
        mean: List of float values used for data standartization. Default: ImageNet mean.
            If there is no need to normalize data, use [0., 0., 0.].
        std: List of float values used for data standartization. Default: ImageNet std.
            If there is no need to normalize data, use [1., 1., 1.].
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
    _weights_url = "https://github.com/photosynthesis-team/" + \
        "photosynthesis.metrics/releases/download/v0.4.0/lpips_weights.pt"

    def __init__(self, replace_pooling: bool = False, distance: str = "mse", reduction: str = "mean",
                 mean: List[float] = IMAGENET_MEAN, std: List[float] = IMAGENET_STD,) -> None:
        lpips_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']
        lpips_weights = torch.hub.load_state_dict_from_url(self._weights_url, progress=False)
        super().__init__("vgg16", layers=lpips_layers, weights=lpips_weights,
                         replace_pooling=replace_pooling, distance=distance,
                         reduction=reduction, mean=mean, std=std,
                         normalize_features=True)


class DISTS(ContentLoss):
    r"""Deep Image Structure and Texture Similarity metric.
    Expects input to be in range [0, 1] or normalized with ImageNet statistics into range [-1, 1]
    Args:
        layers: List of strings with layer names. Default: [`relu3_3`]
        reduction: Reduction over samples in batch: "mean"|"sum"|"none"
        mean: List of float values used for data standartization. Default: ImageNet mean.
            If there is no need to normalize data, use [0., 0., 0.].
        std: List of float values used for data standartization. Default: ImageNet std.
            If there is no need to normalize data, use [1., 1., 1.].
    References:
        .. [1] Keyan Ding, Kede Ma, Shiqi Wang, Eero P. Simoncelli
        (2020). Image Quality Assessment: Unifying Structure and Texture Similarity.
        https://arxiv.org/abs/2004.07728
        .. [2] https://github.com/dingkeyan93/DISTS
    """
    _weights_url = "https://github.com/photosynthesis-team/piq/releases/download/v0.4.1/dists_weights.pt"

    def __init__(self, reduction: str = "mean", mean: List[float] = IMAGENET_MEAN,
                 std: List[float] = IMAGENET_STD) -> None:
        dists_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']
        channels = [3, 64, 128, 256, 512, 512]

        weights = torch.hub.load_state_dict_from_url(self._weights_url, progress=False)
        dists_weights = list(torch.split(weights['alpha'], channels, dim=1))
        dists_weights.extend(torch.split(weights['beta'], channels, dim=1))

        super().__init__("vgg16", layers=dists_layers, weights=dists_weights,
                         replace_pooling=True, reduction=reduction,
                         mean=mean, std=std, normalize_features=False)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = super().forward(prediction, target)
        return 1 - loss

    def compute_distance(self, prediction_features: torch.Tensor, target_features: torch.Tensor) -> List[torch.Tensor]:
        r"""Compute structure similarity between feature maps"""
        structure_distance, texture_distance = [], []
        # Small constant for numerical stability
        EPS = 1e-6

        for x, y in zip(prediction_features, target_features):
            x_mean = x.mean([2, 3], keepdim=True)
            y_mean = y.mean([2, 3], keepdim=True)
            structure_distance.append(similarity_map(x_mean, y_mean, constant=EPS))

            x_var = ((x - x_mean) ** 2).mean([2, 3], keepdim=True)
            y_var = ((y - y_mean) ** 2).mean([2, 3], keepdim=True)
            xy_cov = (x * y).mean([2, 3], keepdim=True) - x_mean * y_mean
            texture_distance.append((2 * xy_cov + EPS) / (x_var + y_var + EPS))

        return structure_distance + texture_distance

    def get_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = super().get_features(x)

        # Add input tensor as an additional feature
        features.insert(0, x)
        return features

    def replace_pooling(self, module: torch.nn.Module) -> torch.nn.Module:
        r"""Turn All MaxPool layers into L2Pool"""
        module_output = module
        if isinstance(module, torch.nn.MaxPool2d):
            module_output = L2Pool2d(kernel_size=3, stride=2, padding=1)
            
        for name, child in module.named_children():
            module_output.add_module(name, self.replace_pooling(child))
        return module_output
