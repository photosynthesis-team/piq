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
    clipiqa = clipiqa.to(device)
    clipiqa(x_grey.to(device))


def test_clip_iqa_fails_with_gray_channels_first(clipiqa: _Loss, x_grey: torch.Tensor, device: str) -> None:
    clipiqa = clipiqa.to(device)
    x_grey = x_grey.permute(0, 2, 3, 1)
    with pytest.raises(AssertionError):
        clipiqa(x_grey.to(device))


def test_clip_iqa_works_with_rgb_channels_last(clipiqa: _Loss, x_rgb: torch.Tensor, device: str) -> None:
    clipiqa = clipiqa.to(device)
    clipiqa(x_rgb.to(device))


def test_clip_iqa_fails_with_rgb_channels_first(clipiqa: _Loss, x_rgb: torch.Tensor, device: str) -> None:
    clipiqa = clipiqa.to(device)
    x_rgb = x_rgb.permute(0, 2, 3, 1)
    with pytest.raises(AssertionError):
        clipiqa(x_rgb.to(device))


def test_clip_iqa_values_rgb(clipiqa: _Loss, device: str) -> None:
    """Reference values are obtained by running the following script on the selected images:
    https://github.com/IceClear/CLIP-IQA/blob/v2-3.8/demo/clipiqa_single_image_demo.py
    """
    clipiqa = clipiqa.to(device)
    paths_scores = {'tests/assets/i01_01_5.bmp': 0.45898438,
                    'tests/assets/I01.BMP': 0.89160156}
    for path, of_score in paths_scores.items():
        img = Image.open(path)
        x_rgb = PILToTensor()(img)
        x_rgb = x_rgb.float()[None]
        score = clipiqa(x_rgb.to(device))
        score_official = torch.tensor([of_score], dtype=torch.float, device=device)
        assert torch.isclose(score, score_official, rtol=1e-2), \
            f'Expected values to be equal to baseline, got {score.item()} and {score_official}'


def test_clip_iqa_input_dtype_does_not_change(clipiqa: _Loss, x_rgb: torch.Tensor, device: str) -> None:
    clipiqa = clipiqa.to(device)
    x_rgb = x_rgb[0][None]
    optional_data_types = torch.float16, torch.float64

    for op_type in optional_data_types:
        x_rgb = x_rgb.type(op_type).to(device)
        clipiqa(x_rgb)
        assert x_rgb.dtype == op_type, \
            f'Expect {op_type} dtype to be preserved, got {x_rgb.dtype}'


def test_clip_iqa_dims_work(clipiqa: _Loss, device: str) -> None:
    clipiqa = clipiqa.to(device)

    x_4dims = [torch.rand((3, 3, 96, 96)), torch.rand((4, 3, 128, 128)), torch.rand((5, 3, 160, 160))]
    for x in x_4dims:
        clipiqa(x.to(device))


def test_clip_iqa_dims_does_not_work(clipiqa: _Loss, device: str) -> None:
    clipiqa = clipiqa.to(device)
    x_2dims = [torch.rand((96, 96)), torch.rand((128, 128)), torch.rand((160, 160))]
    for x in x_2dims:
        with pytest.raises(AssertionError):
            clipiqa(x.to(device))

    x_1dims = [torch.rand((96)), torch.rand((128)), torch.rand((160))]

    for x in x_1dims:
        with pytest.raises(AssertionError):
            clipiqa(x.to(device))

    x_3dims = [torch.rand((3, 96, 96)), torch.rand((3, 128, 128)), torch.rand((3, 160, 160))]
    for x in x_3dims:
        with pytest.raises(AssertionError):
            clipiqa(x.to(device))

    x_5dims = [torch.rand((1, 3, 3, 96, 96)), torch.rand((2, 4, 3, 128, 128)), torch.rand((1, 5, 3, 160, 160))]

    for x in x_5dims:
        with pytest.raises(AssertionError):
            clipiqa(x.to(device))
