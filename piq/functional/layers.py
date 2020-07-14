r"""Custom layers used in metrics computations"""
import torch

from piq.functional import hann_filter


class L2Pool2d(torch.nn.Module):
    r"""Applies L2 pooling with Hann window of size 3x3
    Args:
        x: Tensor with shape (N, C, H, W)"""
    EPS = 1e-12

    def __init__(self, kernel_size: int = 3, stride: int = 2, padding=1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.kernel = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel is None:
            C = x.size(1)
            self.kernel = hann_filter(self.kernel_size).repeat((C, 1, 1, 1)).to(x)

        out = torch.nn.functional.conv2d(
            x ** 2, self.kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.shape[1]
        )
        return (out + self.EPS).sqrt()
