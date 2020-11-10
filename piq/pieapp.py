"""
Implementation of PieAPP
References:
    .. [1] Ekta Prashnani, Hong Cai, Yasamin Mostofi, Pradeep Sen
    (2018). PieAPP: Perceptual Image-Error Assessment through Pairwise Preference
    https://arxiv.org/abs/1806.02067
"""
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from piq.utils import _validate_input, _adjust_dimensions
from piq.functional import crop_patches


class PieAPPModel(nn.Module):
    """Model used for PieAPP score computation
    Args:
        num_feautes: Base feature size, which is multiplied by 2 every 2 blocks
    """
    def __init__(self, features=64):
        super().__init__()

        self.features = features
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten(start_dim=1)

        self.conv1 = nn.Conv2d(3, features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(features, features * 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(features * 2, features * 2, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(features * 2, features * 2, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(features * 2, features * 4, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(features * 4, features * 4, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(features * 4, features * 4, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(features * 4, features * 8, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(features * 8, features * 8, kernel_size=3, padding=1)

        self.fc1_score = nn.Linear(in_features=120832, out_features=512, bias=True)
        self.fc2_score = nn.Linear(in_features=512, out_features=1, bias=True)
        self.fc1_weight = nn.Linear(in_features=2048, out_features=512)
        self.fc2_weight = nn.Linear(in_features=512, out_features=1, bias=True)
        self.ref_score_subtract = nn.Linear(in_features=1, out_features=1, bias=True)

        # Term for numerical stability
        self.EPS = 1e-6

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        r"""
        Returns:
            ... add later
        """
        assert x.shape[2] == x.shape[3] == self.features, \
            f"Expected square input with shape {self.features, self.features}"

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

    def compute_difference(self, features_diff, weights_diff):
        r"""
        Args:
            features_diff: Tensor of shape (N * NUM_PATCHES, C_1)
            features_diff: Tensor of shape (N * NUM_PATCHES, C_2)
        Returns:
            distances
            weights
        """
        # Get scores: fc1_score -> relu -> fc2_score
        # 0.01 is the sigmoid coefficient
        distances = self.ref_score_subtract(0.01 * self.fc2_score(F.relu(self.fc1_score(features_diff))))
#         print("scores", scores.shape)

        weights = self.fc2_weight(F.relu(self.fc1_weight(weights_diff))) + self.EPS
#         print("weights", weights.shape)
        return distances, weights


class PieAPP(_Loss):
    r"""

    Expects input to be in range [0, 1] with no normalization.

    Args:
        reduction: Reduction over samples in batch: "mean"|"sum"|"none"
        stride: ...
        replace_pooling: Flag to replace MaxPooling layer with AveragePooling. See [3] for details. EXPERIMETNAL

    References:
        .. [1] Ekta Prashnani, Hong Cai, Yasamin Mostofi, Pradeep Sen
        (2018). PieAPP: Perceptual Image-Error Assessment through Pairwise Preference
        https://arxiv.org/abs/1806.02067
        .. [2] https://github.com/prashnani/PerceptualImageError
        .. [3] Gatys, Leon and Ecker, Alexander and Bethge, Matthias
        (2016). A Neural Algorithm of Artistic Style}
        Association for Research in Vision and Ophthalmology (ARVO)
        https://arxiv.org/abs/1508.06576
    """
    # TODO: Load weights to release and change this link
    _weights_url = "https://web.ece.ucsb.edu/~ekta/projects/PieAPPv0.1/weights/PieAPPv0.1.pth"

    def __init__(
        self,
        reduction: str = "mean",
        data_range: Union[int, float] = 1.0,
        stride: int = 32,
        replace_pooling: bool = False
    ) -> None:
        super().__init__()
        
        # Load weights and initialize model
        weights = torch.hub.load_state_dict_from_url(self._weights_url, progress=False)
        # Fix small bug in original weights
        weights['ref_score_subtract.weight'] = weights['ref_score_subtract.weight'].unsqueeze(1)
        self.model = PieAPPModel(features=64)
#         print(self.model)
        self.model.load_state_dict(weights)

        if replace_pooling:
            self.model = self.replace_pooling(self.model)

        # Disable gradients
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.data_range = data_range
        self.reduction = reduction
        self.stride = stride

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""Computation of PieAPP  between feature representations of prediction and target tensors.
        Args:
            prediction: Tensor with shape (H, W), (C, H, W) or (N, C, H, W).
            target: Tensor with shape (H, W), (C, H, W) or (N, C, H, W).
        """
        _validate_input(input_tensors=(prediction, target), allow_5d=False, allow_negative=True)
        prediction, target = _adjust_dimensions(input_tensors=(prediction, target))

        N, C, _, _ = prediction.shape

        self.model.to(prediction)
        prediction_features, prediction_weights = self.get_features(prediction)
        target_features, target_weights = self.get_features(target)

        distances, weights = self.model.compute_difference(
            target_features - prediction_features,
            target_weights - prediction_weights
        )

        # Shape (N, NUM_PATCHES)
        distances = distances.reshape(N, -1)
        weights = weights.reshape(N, -1)

        # Scale scores, then average across patches
        loss = torch.stack([(d * w).sum() / w.sum() for d, w in zip(distances, weights)])

        if self.reduction == 'none':
            return loss

        return {'mean': loss.mean,
                'sum': loss.sum
                }[self.reduction](dim=0)

    def get_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        r"""
        Args:
            x: Tensor with shape (N, C, H, W)
        
        Returns:
            features: List of features extracted from intermediate layers
        """
        # Scale input
        x = x * 255 / float(self.data_range)

        x_patches = crop_patches(x, size=64, stride=self.stride)

        features, weights = self.model(x_patches)
        # TODO: Optionally normalize features
        return features, weights

    def replace_pooling(self, module: torch.nn.Module) -> torch.nn.Module:
        r"""Turn All MaxPool layers into AveragePool"""
        module_output = module
        if isinstance(module, torch.nn.MaxPool2d):
            module_output = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            
        for name, child in module.named_children():
            module_output.add_module(name, self.replace_pooling(child))
        return module_output
