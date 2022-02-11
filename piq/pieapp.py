"""
Implementation of PieAPP
References:
    .. [1] Ekta Prashnani, Hong Cai, Yasamin Mostofi, Pradeep Sen
    (2018). PieAPP: Perceptual Image-Error Assessment through Pairwise Preference
    https://arxiv.org/abs/1806.02067
"""
import warnings
from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from piq.utils import _validate_input, _reduce
from piq.functional import crop_patches


class PieAPPModel(nn.Module):
    r""" Model used for PieAPP score computation """
    # Base feature size, which is multiplied by 2 every 2 blocks
    FEATURES = 64

    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten(start_dim=1)

        self.conv1 = nn.Conv2d(3, self.FEATURES, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.FEATURES, self.FEATURES, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(self.FEATURES, self.FEATURES, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(self.FEATURES, self.FEATURES * 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(self.FEATURES * 2, self.FEATURES * 2, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(self.FEATURES * 2, self.FEATURES * 2, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(self.FEATURES * 2, self.FEATURES * 4, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(self.FEATURES * 4, self.FEATURES * 4, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(self.FEATURES * 4, self.FEATURES * 4, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(self.FEATURES * 4, self.FEATURES * 8, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(self.FEATURES * 8, self.FEATURES * 8, kernel_size=3, padding=1)

        # TODO: Reconsider this (hardcoded) implementation as soon as dataset used for PieAPP model training is released
        # Check out project repo: https://github.com/prashnani/PerceptualImageError
        # and project web site http://civc.ucsb.edu/graphics/Papers/CVPR2018_PieAPP/
        # for updates on that.
        self.fc1_score = nn.Linear(in_features=120832, out_features=512, bias=True)
        self.fc2_score = nn.Linear(in_features=512, out_features=1, bias=True)
        self.fc1_weight = nn.Linear(in_features=2048, out_features=512)
        self.fc2_weight = nn.Linear(in_features=512, out_features=1, bias=True)
        self.ref_score_subtract = nn.Linear(in_features=1, out_features=1, bias=True)

        # Term for numerical stability
        self.EPS = 1e-6

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Forward pass a batch of square patches with shape :math:`(N, C, F, F)`.

        Returns:
            features: Concatenation of model features from different scales
            x11: Outputs of the last convolutional layer used as weights
        """
        _validate_input([x, ], dim_range=(4, 4), data_range=(0, -1))

        assert x.shape[2] == x.shape[3] == self.FEATURES, \
            f"Expected square input with shape {self.FEATURES, self.FEATURES}, got {x.shape}"

        # conv1 -> relu -> conv2 -> relu -> pool -> conv3 -> relu
        x3 = F.relu(self.conv3(self.pool(F.relu(self.conv2(F.relu(self.conv1(x)))))))
        # conv4 -> relu -> pool -> conv5 -> relu
        x5 = F.relu(self.conv5(self.pool(F.relu(self.conv4(x3)))))
        # conv6 -> relu -> pool -> conv7 -> relu
        x7 = F.relu(self.conv7(self.pool(F.relu(self.conv6(x5)))))
        # conv8 -> relu -> pool -> conv9 -> relu
        x9 = F.relu(self.conv9(self.pool(F.relu(self.conv8(x7)))))
        # conv10 -> relu -> pool1-> conv11 -> relU
        x11 = self.flatten(F.relu(self.conv11(self.pool(F.relu(self.conv10(x9))))))
        # flatten and concatenate
        features = torch.cat((self.flatten(x3), self.flatten(x5), self.flatten(x7), self.flatten(x9), x11), dim=1)
        return features, x11

    def compute_difference(self, features_diff: torch.Tensor, weights_diff: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            features_diff: Tensor. Shape :math:`(N, C_1)`
            weights_diff: Tensor. Shape :math:`(N, C_2)`

        Returns:
            distances
            weights
        """
        # Get scores: fc1_score -> relu -> fc2_score
        # 0.01 is the sigmoid coefficient
        distances = self.ref_score_subtract(0.01 * self.fc2_score(F.relu(self.fc1_score(features_diff))))

        weights = self.fc2_weight(F.relu(self.fc1_weight(weights_diff))) + self.EPS
        return distances, weights


class PieAPP(_Loss):
    r"""
    Implementation of Perceptual Image-Error Assessment through Pairwise Preference.

    Expects input to be in range ``[0, data_range]`` with no normalization and RGB channel order.
    Input images are cropped into smaller patches. Score for each individual image is mean of it's patch scores.

    Args:
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        data_range: Maximum value range of images (usually 1.0 or 255).
        stride: Step between cropped patches. Smaller values lead to better quality,
            but cause higher memory consumption. Default: 27 (`sparse` sampling in original implementation)
        enable_grad: Flag to compute gradients. Useful when PieAPP used as a loss. Default: False.

    Examples:
        >>> loss = PieAPP()
        >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> y = torch.rand(3, 3, 256, 256)
        >>> output = loss(x, y)
        >>> output.backward()

    References:
        Ekta Prashnani, Hong Cai, Yasamin Mostofi, Pradeep Sen (2018).
        PieAPP: Perceptual Image-Error Assessment through Pairwise Preference
        https://arxiv.org/abs/1806.02067

        https://github.com/prashnani/PerceptualImageError

    """
    _weights_url = "https://github.com/photosynthesis-team/piq/releases/download/v0.5.4/PieAPPv0.1.pth"

    def __init__(self, reduction: str = "mean", data_range: Union[int, float] = 1.0, stride: int = 27,
                 enable_grad: bool = False) -> None:
        super().__init__()

        # Load weights and initialize model
        weights = torch.hub.load_state_dict_from_url(self._weights_url, progress=False)
        # Fix small bug in original weights
        weights['ref_score_subtract.weight'] = weights['ref_score_subtract.weight'].unsqueeze(1)
        self.model = PieAPPModel()
        self.model.load_state_dict(weights)

        # Disable gradients
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.data_range = data_range
        self.reduction = reduction
        self.stride = stride
        self.enable_grad = enable_grad

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""
        Computation of PieAPP  between feature representations of prediction :math:`x` and target :math:`y` tensors.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)`.
            y: A target tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            Perceptual Image-Error Assessment through Pairwise Preference
        """
        _validate_input([x, y], dim_range=(4, 4), data_range=(0, self.data_range))

        N, C, _, _ = x.shape
        if C == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
            warnings.warn('The original PieAPP supports only RGB images.'
                          'The input images were converted to RGB by copying the grey channel 3 times.')

        self.model.to(device=x.device)
        x_features, x_weights = self.get_features(x)
        y_features, y_weights = self.get_features(y)

        distances, weights = self.model.compute_difference(
            y_features - x_features,
            y_weights - x_weights
        )

        distances = distances.reshape(N, -1)
        weights = weights.reshape(N, -1)

        # Scale scores, then average across patches
        loss = torch.stack([(d * w).sum() / w.sum() for d, w in zip(distances, weights)])

        return _reduce(loss, self.reduction)

    def get_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""

        Args:
            x: Tensor. Shape :math:`(N, C, H, W)`.

        Returns:
            List of features extracted from intermediate layers weights
        """
        # Rescale to [0, 255] range on which models was trained
        x = x / float(self.data_range) * 255
        x_patches = crop_patches(x, size=64, stride=self.stride)

        with torch.autograd.set_grad_enabled(self.enable_grad):
            features, weights = self.model(x_patches)

        return features, weights
