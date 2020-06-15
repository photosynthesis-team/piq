"""
Implementation of VGG16 loss, originaly used for style transfer and usefull in many other task (including GAN training)
It's work in progress, no guarantees that code will work
"""
# import collections
from typing import List

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torchvision.models import vgg16, vgg19


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

    Args:
        feature_extractor: Name of model used to extract features. One of {`vgg16`, `vgg19`}
        layers: List of string with layer names. Default: [`relu3_3`]
        weights: List of float weight to balance different layers
        replace_pooling: Flag to replace MaxPooling layer with AveragePooling. See [1] for details.
        distance: Method to compute distance between features. One of {`mse`, `mae`}.
        reduction: Reduction over samples in batch: "mean"|"sum"|"none"


    Uses pretrained VGG16 model from torchvision by default
    layers: list of VGG layers used to evaluate content loss

    reduction: Type of reduction to use
    References:
        .. [1] Gatys, Leon and Ecker, Alexander and Bethge, Matthias
           (2016). A Neural Algorithm of Artistic Style}
           Association for Research in Vision and Ophthalmology (ARVO)
           https://arxiv.org/abs/1508.06576
    """

    def __init__(self, feature_extractor: str = "vgg16", layers: List[str] = ["relu3_3"],
                 weights: List[float] = [1.], replace_pooling: bool = False, distance: str = "mse",
                 reduction: str = "mean") -> None:
        super().__init__()

        if feature_extractor == "vgg16":
            self.model = vgg16(pretrained=True, progress=False)
        elif feature_extractor == "vgg19":
            self.model = vgg19(pretrained=True, progress=False)
        else:
            raise ValueError("Unknown feature extractor")

        if replace_pooling:
            for i, layer in enumerate(self.model.features):
                if isinstance(layer, torch.nn.MaxPool2d):
                    self.model.features[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        # Disable gradients
        for param in self.model.parameters():
            param.requires_grad_(False)

        if distance == "mse":
            self.criterion = nn.MSELoss(reduction=reduction)
        elif distance == "mae":
            self.criterion = nn.L1Loss(reduction=reduction)
        else:
            raise ValueError("Unknown distance. Should be in {`mse`, `mae`}")

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Computation of Content loss between feature representations of prediction and target tensors.

        Args:
            prediction: Tensor of prediction of the network.
            target: Reference tensor.

        """
        input_features = torch.stack(self.get_features(prediction))
        content_features = torch.stack(self.get_features(prediction))
        loss = self.criterion(input_features, content_features)

        # Solve big memory consumption
        torch.cuda.empty_cache()
    
        return loss

    def get_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: torch.Tensor with shape (N, C, H, W)
        
        Returns:
            List of features extracted from intermediate layers
        """

        features = []
        for name, module in self.model.features._modules.items():
            x = module(x)
            if name in self.layers:
                features.append(x)
        # print(len(features))
        return features


class StyleLoss(_Loss):
    """
    Class for creating style loss for neural style transfer
    model: str in ['vgg16_bn']
    reduction: Type of reduction to use
    """

    def __init__(
        self,
        layers=["0", "5", "10", "19", "28"],
        weights=[0.75, 0.5, 0.2, 0.2, 0.2],
        loss="mse",
        device="cuda",
        reduction="mean",
        **args,
    ):
        super().__init__()
        self.model = vgg16(pretrained=True, **args)
        self.model.eval().to(device)

        # self.layers = listify(layers)
        # self.weights = listify(weights)

        if loss == "mse":
            self.criterion = nn.MSELoss(reduction=reduction)
        elif loss == "mae":
            self.criterion = nn.L1Loss(reduction=reduction)
        else:
            raise KeyError

    def forward(self, input, style):
        """
        Measure distance between feature representations of input and content images
        """
        input_features = self.get_features(input)
        style_features = self.get_features(style)
        # print(style_features[0].size(), len(style_features))

        input_gram = [self.gram_matrix(x) for x in input_features]
        style_gram = [self.gram_matrix(x) for x in style_features]

        loss = [
            self.criterion(torch.stack(i_g), torch.stack(s_g)) for i_g, s_g in zip(input_gram, style_gram)
        ]
        return torch.mean(torch.tensor(loss))

    def get_features(self, x):
        """
        Extract feature maps from the intermediate layers.
        """
        if self.layers is None:
            self.layers = ["0", "5", "10", "19", "28"]

        features = []
        for name, module in self.model.features._modules.items():
            x = module(x)
            if name in self.layers:
                features.append(x)
        return features

    def gram_matrix(self, input):
        """
        Compute Gram matrix for each image in batch
        input: Tensor of shape BxCxHxW
            B: batch size
            C: channels size
            H&W: spatial size
        """

        B, C, H, W = input.size()
        gram = []
        for i in range(B):
            x = input[i].view(C, H * W)
            gram.append(torch.mm(x, x.t()))
        return gram
