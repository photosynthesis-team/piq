
class DISTS(ContentLoss):
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
                 std: List[float] = IMAGENET_STD) -> None:
        dists_layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3']
        channels = [3, 64, 128, 256, 512, 512]

        weights = torch.hub.load_state_dict_from_url(self._weights_url, progress=False)
        dists_weights = list(torch.split(weights['alpha'], channels, dim=1))
        dists_weights.extend(torch.split(weights['beta'], channels, dim=1))

        super().__init__("vgg16", layers=dists_layers, weights=dists_weights,
                         replace_pooling=True, reduction=reduction, mean=mean, std=std,
                         normalize_features=False, allow_layers_weights_mismatch=True)

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

        loss = super().forward(x, y)
        return 1 - loss

    def compute_distance(self, x_features: torch.Tensor, y_features: torch.Tensor) -> List[torch.Tensor]:
        r"""Compute structure similarity between feature maps

        Args:
            x_features: Features of the input tensor.
            y_features: Features of the target tensor.

        Returns:
            Structural similarity distance between feature maps
        """
        structure_distance, texture_distance = [], []
        # Small constant for numerical stability
        EPS = 1e-6

        for x, y in zip(x_features, y_features):
            x_mean = x.mean([2, 3], keepdim=True)
            y_mean = y.mean([2, 3], keepdim=True)
            structure_distance.append(similarity_map(x_mean, y_mean, constant=EPS))

            x_var = ((x - x_mean) ** 2).mean([2, 3], keepdim=True)
            y_var = ((y - y_mean) ** 2).mean([2, 3], keepdim=True)
            xy_cov = (x * y).mean([2, 3], keepdim=True) - x_mean * y_mean
            texture_distance.append((2 * xy_cov + EPS) / (x_var + y_var + EPS))

        return structure_distance + texture_distance

    def get_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        r"""

        Args:
            x: Input tensor

        Returns:
            List of features extracted from input tensor
        """
        features = super().get_features(x)

        # Add input tensor as an additional feature
        features.insert(0, x)
        return features

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
