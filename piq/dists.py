
from typing import List

import torch
from torch.nn.modules.loss import _Loss
import torchvision

from piq.utils import _validate_input, _reduce
from piq.functional import similarity_map, L2Pool2d
from piq.perceptual import VGG16_LAYERS, IMAGENET_MEAN, IMAGENET_STD


class DISTS(_Loss):
    r"""Deep Image Structure and Texture Similarity metric.

    By default expects input to be in range [0, 1], which is then normalized by ImageNet statistics into range [-1, 1].
    If no normalisation is required, change `mean` and `std` values accordingly.

    Args:
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        mean: List of float values used for data standardization. Default: ImageNet mean.
            If there is no need to normalize data, use [0., 0., 0.].
        std: List of float values used for data standardization. Default: ImageNet std.
            If there is no need to normalize data, use [1., 1., 1.].
        enable_grad: Enable gradient computation. Default: ``False``

    Examples:
        >>> loss = DISTS()
        >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> y = torch.rand(3, 3, 256, 256)
        >>> output = loss(x, y)
        >>> output.backward()

    References:
        Keyan Ding, Kede Ma, Shiqi Wang, Eero P. Simoncelli (2020).
        Image Quality Assessment: Unifying Structure and Texture Similarity.
        https://arxiv.org/abs/2004.07728
        https://github.com/dingkeyan93/DISTS
    """
    _weights_url = "https://github.com/photosynthesis-team/piq/releases/download/v0.4.1/dists_weights.pt"

    def __init__(self, reduction: str = "mean", mean: List[float] = IMAGENET_MEAN,
                 std: List[float] = IMAGENET_STD, enable_grad: bool = False) -> None:
        super().__init__()

        dists_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']
        channels = [3, 64, 128, 256, 512, 512]

        weights = torch.hub.load_state_dict_from_url(self._weights_url, progress=False)
        dists_weights = list(torch.split(weights['alpha'], channels, dim=1))
        dists_weights.extend(torch.split(weights['beta'], channels, dim=1))

        self.model = torchvision.models.vgg16(pretrained=True, progress=False).features
        self.layers = [VGG16_LAYERS[l] for l in dists_layers]

        self.model = self.replace_pooling(self.model)

        # Disable gradients
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.weights = [torch.tensor(w) if not isinstance(w, torch.Tensor) else w for w in dists_weights]

        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)
        self.reduction = reduction
        self.enable_grad = enable_grad

        # normalize_features=False, allow_layers_weights_mismatch=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Deep Image Structure and Texture Similarity loss, i.e. ``1-DISTS`` in range [0, 1].
        """
        _, _, H, W = x.shape

        if min(H, W) > 256:
            x = torch.nn.functional.interpolate(
                x, scale_factor=256 / min(H, W), recompute_scale_factor=False, mode='bilinear')
            y = torch.nn.functional.interpolate(
                y, scale_factor=256 / min(H, W), recompute_scale_factor=False, mode='bilinear')

        _validate_input([x, y], dim_range=(4, 4), data_range=(0, -1))

        self.model.to(x)
        self.mean, self.std = self.mean.to(x), self.std.to(x)

        # Normalize
        x, y = (x - self.mean) / self.std, (y - self.mean) / self.std

        # Add input tensor as an additional feature
        x_features, y_features = [x, ], [y, ]
        with torch.autograd.set_grad_enabled(self.enable_grad):
            for name, module in self.model._modules.items():
                x = module(x)
                y = module(y)
                if name in self.layers:
                    x_features.append(x)
                    y_features.append(y)

        # Compute structure similarity between feature maps
        EPS = 1e-6  # Small constant for numerical stability

        structure_distance, texture_distance = [], []
        for x, y in zip(x_features, y_features):
            x_mean = x.mean([2, 3], keepdim=True)
            y_mean = y.mean([2, 3], keepdim=True)
            structure_distance.append(similarity_map(x_mean, y_mean, constant=EPS))

            x_var = ((x - x_mean) ** 2).mean([2, 3], keepdim=True)
            y_var = ((y - y_mean) ** 2).mean([2, 3], keepdim=True)
            xy_cov = (x * y).mean([2, 3], keepdim=True) - x_mean * y_mean
            texture_distance.append((2 * xy_cov + EPS) / (x_var + y_var + EPS))

        distances = structure_distance + texture_distance

        # Scale distances, then average in spatial dimensions, then stack and sum in channels dimension
        loss = torch.cat([(d * w.to(d)).mean(dim=[2, 3]) for d, w in zip(distances, self.weights)], dim=1).sum(dim=1)

        return 1 - _reduce(loss, self.reduction)

    def replace_pooling(self, module: torch.nn.Module) -> torch.nn.Module:
        r"""Turn All MaxPool layers into L2Pool

        Args:
            module: Module to change MaxPool into L2Pool

        Returns:
            Module with L2Pool instead of MaxPool
        """
        module_output = module
        if isinstance(module, torch.nn.MaxPool2d):
            module_output = L2Pool2d(kernel_size=3, stride=2, padding=1)

        for name, child in module.named_children():
            module_output.add_module(name, self.replace_pooling(child))

        return module_output
