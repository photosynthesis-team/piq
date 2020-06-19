r""" This module implements InceptionNetV3 for computation of Frechet Inception Distance (FID) in PyTorch.

Implementation of classes and functions from this module are inspired by @mseitzer's implementation.
@mseitzer's implementation is licenced under Apache-2.0 licence:
https://github.com/mseitzer/pytorch-fid
"""

import torch

import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from typing import List

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# Inception weights ported to PyTorch from
# http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
FID_WEIGHTS_URL = \
    'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'


class InceptionV3(nn.Module):
    r"""Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling.
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices.
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features.
        192: 1,  # Second max pooling features.
        768: 2,  # Pre-aux classifier features.
        2048: 3  # Final average pooling features.
    }

    def __init__(self,
                 output_blocks: List[int] = [DEFAULT_BLOCK_INDEX],
                 resize_input: bool = True,
                 normalize_input: bool = True,
                 requires_grad: bool = False,
                 use_fid_inception: bool = True) -> None:
        r"""Build pretrained InceptionV3

        Args:
            output_blocks: Indices of blocks to return features of. Possible values are:
                    - 0: corresponds to output of first max pooling
                    - 1: corresponds to output of second max pooling
                    - 2: corresponds to output which is fed to aux classifier
                    - 3: corresponds to output of final average pooling
            resize_input:  If true, bilinearly resizes input to width and height 299 before
                feeding input to model. As the network without fully connected
                layers is fully convolutional, it should be able to handle inputs
                of arbitrary size, so resizing might not be strictly needed
            normalize_input: If true, scales the input from range (0, 1) to the range the
                pretrained Inception network expects, namely (-1, 1)
            requires_grad: If true, parameters of the model require gradients.
                Possibly useful for finetuning the network
            use_fid_inception: If true, uses the pretrained Inception model used in Tensorflow's
                FID implementation. If false, uses the pretrained Inception model
                available in torchvision. The FID Inception model has different
                weights and a slightly different structure from torchvision's
                Inception model. If you want to compute FID scores, you are
                strongly advised to set this parameter to true to get comparable
                results.
        """
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        if use_fid_inception:
            inception = self.fid_inception_v3()
        else:
            inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1.
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2.
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier.
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool.
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp: torch.autograd.Variable) -> List[torch.autograd.Variable]:
        r"""Get Inception feature maps

        Args:
            inp: Input tensor of shape Bx3xHxW. Values are expected to be in range (0, 1).

        Returns:
            List of torch.autograd.Variable, corresponding to the selected output block, sorted ascending by index.
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            # Scale from range (0, 1) to range (-1, 1).
            x = 2 * x - 1

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp

    @staticmethod
    def fid_inception_v3() -> nn.Module:
        r"""Build pretrained Inception model for FID computation

        The Inception model for FID computation uses a different set of weights
        and has a slightly different structure than torchvision's Inception.
        This method first constructs torchvision's Inception and then patches the
        necessary parts that are different in the FID Inception model.
        """
        inception = models.inception_v3(num_classes=1008,
                                        aux_logits=False,
                                        pretrained=False)
        inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
        inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
        inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
        inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
        inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
        inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
        inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
        inception.Mixed_7b = FIDInceptionE1(1280)
        inception.Mixed_7c = FIDInceptionE2(2048)

        state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
        inception.load_state_dict(state_dict)

        return inception


class FIDInceptionA(models.inception.InceptionA):
    r"""InceptionA block patched for FID computation."""
    def __init__(self, in_channels: int, pool_features: int) -> None:
        super(FIDInceptionA, self).__init__(in_channels, pool_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation.
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionC(models.inception.InceptionC):
    r"""InceptionC block patched for FID computation."""
    def __init__(self, in_channels: int, channels_7x7: int) -> None:
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation.
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE1(models.inception.InceptionE):
    r"""First InceptionE block patched for FID computation."""
    def __init__(self, in_channels: int) -> None:
        super(FIDInceptionE1, self).__init__(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: Tensorflow's average pool does not use the padded zero's in
        # its average calculation.
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                   count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class FIDInceptionE2(models.inception.InceptionE):
    r"""Second InceptionE block patched for FID computation."""
    def __init__(self, in_channels: int) -> None:
        super(FIDInceptionE2, self).__init__(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        # Patch: The FID Inception model uses max pooling instead of average
        # pooling. This is likely an error in this specific Inception
        # implementation, as other Inception models use average pooling here
        # (which matches the description in the paper).
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)
