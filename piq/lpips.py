"""
Implementation of Learned Perceptual Image Patch Similarity (LPIPS) metric
References:
    Zhang, Richard and Isola, Phillip and Efros, et al. (2018)
    The Unreasonable Effectiveness of Deep Features as a Perceptual Metric
    IEEE/CVF Conference on Computer Vision and Pattern Recognition
    https://arxiv.org/abs/1801.03924
    https://github.com/richzhang/PerceptualSimilarity
"""

from typing import List

import torch
import torchvision
import torch.nn as nn
from torch.nn.modules.loss import _Loss

from piq.utils import _validate_input, _reduce
from piq.perceptual import VGG16_LAYERS, IMAGENET_MEAN, IMAGENET_STD, EPS


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
        enable_grad: Flag to compute gradients. Useful when LPIPS used as a loss. Default: False.

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
                 mean: List[float] = IMAGENET_MEAN, std: List[float] = IMAGENET_STD,
                 enable_grad: bool = False) -> None:
        super().__init__()

        lpips_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']
        lpips_weights = torch.hub.load_state_dict_from_url(self._weights_url, progress=False)

        self.model = torchvision.models.vgg16(pretrained=True, progress=False).features
        self.layers = [VGG16_LAYERS[l] for l in lpips_layers]

        if replace_pooling:
            self.model = self.replace_pooling(self.model)

        # Disable gradients
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.distance = {
            "mse": nn.MSELoss,
            "mae": nn.L1Loss,
        }[distance](reduction='none')

        self.weights = [torch.tensor(w) if not isinstance(w, torch.Tensor) else w for w in lpips_weights]

        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)
        self.reduction = reduction
        self.enable_grad = enable_grad

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""LPIPS computation between :math:`x` and :math:`y` tensors.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            LPIPS value between inputs.
        """
        _validate_input([x, y], dim_range=(4, 4), data_range=(0, -1))

        self.model.to(x)
        self.mean, self.std = self.mean.to(x), self.std.to(x)

        # Normalize
        x, y = (x - self.mean) / self.std, (y - self.mean) / self.std

        x_features, y_features = [], []
        with torch.autograd.set_grad_enabled(self.enable_grad):
            for name, module in self.model._modules.items():
                x = module(x)
                y = module(y)
                if name in self.layers:
                    # Normalize feature maps in channel direction to unit length.
                    x_norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
                    y_norm_factor = torch.sqrt(torch.sum(y ** 2, dim=1, keepdim=True))

                    x_features.append(x / (x_norm_factor + EPS))
                    y_features.append(y / (y_norm_factor + EPS))

        distances = [self.distance(x, y) for x, y in zip(x_features, y_features)]

        # Scale distances, then average in spatial dimensions, then stack and sum in channels dimension
        loss = torch.cat([(d * w.to(d)).mean(dim=[2, 3]) for d, w in zip(distances, self.weights)], dim=1).sum(dim=1)

        return _reduce(loss, self.reduction)

    def replace_pooling(self, module: torch.nn.Module) -> torch.nn.Module:
        r"""Turn all MaxPool layers into AveragePool

        Args:
            module: Module to change MaxPool int AveragePool

        Returns:
            Module with AveragePool instead MaxPool

        """
        module_output = module
        if isinstance(module, torch.nn.MaxPool2d):
            module_output = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        for name, child in module.named_children():
            module_output.add_module(name, self.replace_pooling(child))
        return module_output
