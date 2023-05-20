import torch
import pytest

from PIL import Image
from piq import CLIPIQA
from torchvision.transforms import PILToTensor
from torch.nn.modules.loss import _Loss


@pytest.fixture(scope='module')
def x_grey() -> torch.Tensor:
    return torch.rand(3, 1, 96, 96)


@pytest.fixture(scope='module')
def x_rgb() -> torch.Tensor:
    return torch.rand(3, 3, 96, 96)


@pytest.fixture(scope='module')
def clipiqa() -> _Loss:
    return CLIPIQA(data_range=255)


# ================== Test class: `CLIPIQA` ==================
def test_clip_iqa_works_with_grey_channels_last(clipiqa: _Loss, x_grey: torch.Tensor, device: str) -> None:
    x_grey = x_grey.permute(0, 2, 3, 1)
    clipiqa(x_grey.to(device))


def test_clip_iqa_works_with_rgb_channels_last(clipiqa: _Loss, x_grey: torch.Tensor, device: str) -> None:
    x_grey = x_grey.permute(0, 2, 3, 1)
    clipiqa(x_grey.to(device))


def test_clip_iqa_works_with_grey_channels_first(clipiqa: _Loss, x_grey: torch.Tensor, device: str) -> None:
    clipiqa(x_grey.to(device))


def test_clip_iqa_works_with_rgb_channels_first(clipiqa: _Loss, x_grey: torch.Tensor, device: str) -> None:
    clipiqa(x_grey.to(device))


def test_clip_iqa_values_rgb(clipiqa: _Loss, device: str) -> None:
    """Reference values are obtained by running the following script on the selected images:
    https://github.com/IceClear/CLIP-IQA/blob/v2-3.8/demo/clipiqa_single_image_demo.py
    """
    paths_scores = {'tests/assets/i01_01_5.bmp': 0.45898438,
                    'tests/assets/I01.BMP': 0.89160156}
    for path, of_score in paths_scores.items():
        img = Image.open(path)
        x_rgb = PILToTensor()(img).permute(1, 2, 0).float()[None]
        print('x_rgb.min(), x_rgb.max()', x_rgb.min(), x_rgb.max())
        print('x_rgb.shape', x_rgb.shape)
        score = clipiqa(x_rgb.to(device))
        score_official = torch.tensor([of_score], dtype=torch.float, device=device)
        assert torch.isclose(score, score_official, rtol=1e-2), \
            f'Expected values to be equal to baseline, got {score.item()} and {score_official}'
