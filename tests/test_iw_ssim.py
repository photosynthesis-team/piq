import torch
import itertools
import pytest
from piq import InformationWeightedSSIMLoss, information_weighted_ssim
from typing import Tuple, List
from skimage.io import imread
from contextlib import contextmanager


@contextmanager
def raise_nothing(enter_result=None):
    yield enter_result


@pytest.fixture(scope='module')
def x_rand() -> torch.Tensor:
    return torch.rand(3, 3, 161, 161)


@pytest.fixture(scope='module')
def y_rand() -> torch.Tensor:
    return torch.rand(3, 3, 161, 161)


@pytest.fixture(scope='module')
def ones_zeros_4d() -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ones(3, 3, 161, 161), torch.zeros(3, 3, 161, 161)


@pytest.fixture(scope='module')
def test_images() -> List[Tuple[torch.Tensor, torch.Tensor]]:
    x_grey = torch.tensor(imread('tests/assets/goldhill_jpeg.gif')).unsqueeze(0).unsqueeze(0)
    y_grey = torch.tensor(imread('tests/assets/goldhill.gif')).unsqueeze(0).unsqueeze(0)
    x_rgb = torch.tensor(imread('tests/assets/I01.BMP')).permute(2, 0, 1).unsqueeze(0)
    y_rgb = torch.tensor(imread('tests/assets/i01_01_5.bmp')).permute(2, 0, 1).unsqueeze(0)
    return [(x_grey, y_grey), (x_rgb, y_rgb)]


@pytest.fixture(scope='module')
def scale_weights() -> torch.Tensor:
    return torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])


# ================== Test function: `information_weighted_ssim` ==================
def test_iw_ssim_measure_is_one_for_equal_tensors(x_rand: torch.Tensor, device: str) -> None:
    x_rand = x_rand.to(device)
    y_rand = x_rand.clone()
    measure = information_weighted_ssim(x_rand, y_rand, data_range=1.)
    assert torch.allclose(measure, torch.ones_like(measure)), \
        f'If equal tensors are passed IW-SSIM must be equal to 1 ' \
        f'(considering floating point operation error up to 1 * 10^-6), got {measure + 1}'


def test_iw_ssim_reduction(x_rand: torch.Tensor, y_rand: torch.Tensor, device: str) -> None:
    for mode in ['mean', 'sum', 'none']:
        information_weighted_ssim(x_rand.to(device), y_rand.to(device), reduction=mode)

    for mode in [None, 'n', 2]:
        with pytest.raises(ValueError):
            information_weighted_ssim(x_rand.to(device), y_rand.to(device), reduction=mode)


def test_iw_ssim_raises_if_tensors_have_different_shapes(x_rand: torch.Tensor, y_rand: torch.Tensor,
                                                         scale_weights: torch.Tensor, device: str) -> None:

    dims = [[3], [2, 3], [160, 161], [160, 161]]

    for size in list(itertools.product(*dims)):
        wrong_shape_x = torch.rand(size).to(x_rand)
        print(wrong_shape_x.size())
        if wrong_shape_x.size() == x_rand.size():
            information_weighted_ssim(wrong_shape_x.to(device), x_rand.to(device))
        else:
            with pytest.raises(AssertionError):
                information_weighted_ssim(wrong_shape_x.to(device), x_rand.to(device))

    information_weighted_ssim(x_rand.to(device), y_rand.to(device), scale_weights=scale_weights.to(device))

    wrong_scale_weights = torch.rand(2, 2)
    with pytest.raises(ValueError):
        information_weighted_ssim(x_rand.to(device), y_rand.to(device), scale_weights=wrong_scale_weights.to(device))


def test_iw_ssim_raises_if_tensors_have_different_types(x_rand: torch.Tensor, device: str) -> None:
    wrong_type_x = list(range(10))
    with pytest.raises(AssertionError):
        information_weighted_ssim(wrong_type_x, x_rand.to(device))


def test_iw_ssim_raises_if_kernel_size_greater_than_image(x_rand: torch.Tensor, y_rand: torch.Tensor,
                                                          device: str) -> None:
    kernel_size = 11
    levels = 5
    min_size = (kernel_size - 1) * 2 ** (levels - 1) + 1
    wrong_size_x = x_rand[:, :, :min_size - 1, :min_size - 1]
    wrong_size_y = y_rand[:, :, :min_size - 1, :min_size - 1]
    with pytest.raises(ValueError):
        information_weighted_ssim(wrong_size_x.to(device), wrong_size_y.to(device), kernel_size=kernel_size)


@pytest.mark.parametrize(
    "data_range", [128, 255],
)
def test_iw_ssim_supports_different_data_ranges(x_rand: torch.Tensor, y_rand: torch.Tensor, data_range: int,
                                                device: str) -> None:

    x_scaled = (x_rand * data_range).type(torch.uint8)
    y_scaled = (y_rand * data_range).type(torch.uint8)

    measure_scaled = information_weighted_ssim(x_scaled.to(device), y_scaled.to(device), data_range=data_range)
    measure = information_weighted_ssim(
        x_scaled.to(device) / float(data_range),
        y_scaled.to(device) / float(data_range),
        data_range=1.0
    )
    diff = torch.abs(measure_scaled - measure)
    assert (diff <= 1e-6).all(), f'Result for same tensor with different data_range should be the same, got {diff}'


def test_iw_ssim_fails_for_incorrect_data_range(x_rand: torch.Tensor, y_rand: torch.Tensor, device: str) -> None:
    # Scale to [0, 255]
    x_scaled = (x_rand * 255).type(torch.uint8)
    y_scaled = (y_rand * 255).type(torch.uint8)
    with pytest.raises(AssertionError):
        information_weighted_ssim(x_scaled.to(device), y_scaled.to(device), data_range=1.0)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64],
)
def test_iw_ssim_preserves_dtype(x_rand: torch.Tensor, y_rand: torch.Tensor, dtype: torch.dtype, device: str) -> None:
    output = information_weighted_ssim(x_rand.to(device=device, dtype=dtype), y_rand.to(device=device, dtype=dtype))
    assert output.dtype == dtype


def test_iw_ssim_corresponds_to_matlab(test_images: List, device: str):
    x_gray, y_gray = test_images[0]
    x_rgb, y_rgb = test_images[1]
    matlab_gray = torch.tensor(0.886297251092821, device=device)
    matlab_rgb = torch.tensor(0.946804801436296, device=device)

    score_gray = information_weighted_ssim(x_gray.to(device), y_gray.to(device), data_range=255)

    assert torch.isclose(score_gray, matlab_gray, atol=1e-5),\
        f'Expected {matlab_gray:.4f}, got {score_gray:.4f} for gray scale case.'

    score_rgb = information_weighted_ssim(x_rgb.to(device), y_rgb.to(device), data_range=255)

    assert torch.isclose(score_rgb, matlab_rgb, atol=1e-5),\
        f'Expected {matlab_rgb:.8f}, got {score_rgb:.8f} for rgb case.'


# ================== Test class: `InformationWeightedSSIMLoss` ==================
def test_iw_ssim_loss_is_one_for_equal_tensors(x_rand: torch.Tensor, device: str) -> None:
    x_rand = x_rand.to(device)
    y_rand = x_rand.clone()
    loss = InformationWeightedSSIMLoss(data_range=1.)
    measure = loss(x_rand, y_rand)
    assert torch.allclose(measure, torch.zeros_like(measure), atol=1e-5), \
        f'If equal tensors are passed IW-SSIM must be equal to 0 ' \
        f'(considering floating point operation error up to 1 * 10^-5), got {measure}'


def test_iw_ssim_loss_reduction(x_rand: torch.Tensor, y_rand: torch.Tensor, device: str) -> None:
    for mode in ['mean', 'sum', 'none']:
        loss = InformationWeightedSSIMLoss(reduction=mode)
        loss(x_rand.to(device), y_rand.to(device))

    for mode in [None, 'n', 2]:
        with pytest.raises(ValueError):
            loss = InformationWeightedSSIMLoss(reduction=mode)
            loss(x_rand.to(device), y_rand.to(device))


def test_iw_ssim_loss_raises_if_tensors_have_different_shapes(x_rand: torch.Tensor, y_rand: torch.Tensor,
                                                              scale_weights: torch.Tensor, device: str) -> None:

    dims = [[3], [2, 3], [160, 161], [160, 161]]
    loss = InformationWeightedSSIMLoss(data_range=1.)
    for size in list(itertools.product(*dims)):
        wrong_shape_x = torch.rand(size).to(x_rand)
        print(wrong_shape_x.size())
        if wrong_shape_x.size() == x_rand.size():
            loss(wrong_shape_x.to(device), x_rand.to(device))
        else:
            with pytest.raises(AssertionError):
                loss(wrong_shape_x.to(device), x_rand.to(device))

    loss = InformationWeightedSSIMLoss(data_range=1., scale_weights=scale_weights.to(device))
    loss(x_rand.to(device), y_rand.to(device))
    wrong_scale_weights = torch.rand(2, 2)
    loss = InformationWeightedSSIMLoss(data_range=1., scale_weights=wrong_scale_weights.to(device))
    with pytest.raises(ValueError):
        loss(x_rand.to(device), y_rand.to(device))


def test_iw_ssim_loss_raises_if_tensors_have_different_types(x_rand: torch.Tensor, device: str) -> None:
    wrong_type_x = list(range(10))
    loss = InformationWeightedSSIMLoss(data_range=1.)
    with pytest.raises(AssertionError):
        loss(wrong_type_x, x_rand.to(device))


def test_iw_ssim_loss_raises_if_kernel_size_greater_than_image(x_rand: torch.Tensor, y_rand: torch.Tensor,
                                                               device: str) -> None:
    kernel_size = 11
    levels = 5
    min_size = (kernel_size - 1) * 2 ** (levels - 1) + 1
    wrong_size_x = x_rand[:, :, :min_size - 1, :min_size - 1]
    wrong_size_y = y_rand[:, :, :min_size - 1, :min_size - 1]
    loss = InformationWeightedSSIMLoss(data_range=1., kernel_size=kernel_size)
    with pytest.raises(ValueError):
        loss(wrong_size_x.to(device), wrong_size_y.to(device))


@pytest.mark.parametrize(
    "data_range", [128, 255],
)
def test_iw_ssim_loss_supports_different_data_ranges(x_rand: torch.Tensor, y_rand: torch.Tensor, data_range: int,
                                                     device: str) -> None:

    x_scaled = (x_rand * data_range).type(torch.uint8)
    y_scaled = (y_rand * data_range).type(torch.uint8)

    loss = InformationWeightedSSIMLoss(data_range=1.)
    loss_scaled = InformationWeightedSSIMLoss(data_range=data_range)

    measure_scaled = loss_scaled(x_scaled.to(device), y_scaled.to(device))
    measure = loss(
        x_scaled.to(device) / float(data_range),
        y_scaled.to(device) / float(data_range)
    )
    diff = torch.abs(measure_scaled - measure)
    assert (diff <= 1e-6).all(), f'Result for same tensor with different data_range should be the same, got {diff}'


def test_iw_ssim_loss_fails_for_incorrect_data_range(x_rand: torch.Tensor, y_rand: torch.Tensor, device: str) -> None:
    # Scale to [0, 255]
    x_scaled = (x_rand * 255).type(torch.uint8)
    y_scaled = (y_rand * 255).type(torch.uint8)
    loss = InformationWeightedSSIMLoss(data_range=1.)
    with pytest.raises(AssertionError):
        loss(x_scaled.to(device), y_scaled.to(device))


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64],
)
def test_iw_ssim_loss_preserves_dtype(x_rand: torch.Tensor, y_rand: torch.Tensor, dtype: torch.dtype,
                                      device: str) -> None:
    loss = InformationWeightedSSIMLoss(data_range=1.)
    output = loss(x_rand.to(device=device, dtype=dtype), y_rand.to(device=device, dtype=dtype))
    assert output.dtype == dtype


def test_iw_ssim_loss_corresponds_to_matlab(test_images: List, device: str):
    x_gray, y_gray = test_images[0]
    x_rgb, y_rgb = test_images[1]
    matlab_gray = 1 - torch.tensor(0.886297251092821, device=device)
    matlab_rgb = 1 - torch.tensor(0.946804801436296, device=device)

    loss = InformationWeightedSSIMLoss(data_range=255)
    score_gray = loss(x_gray.to(device), y_gray.to(device))

    assert torch.isclose(score_gray, matlab_gray, atol=1e-5),\
        f'Expected {matlab_gray:.8f}, got {score_gray:.8f} for gray scale case.'

    score_rgb = loss(x_rgb.to(device), y_rgb.to(device))

    assert torch.isclose(score_rgb, matlab_rgb, atol=1e-5),\
        f'Expected {matlab_rgb:.8f}, got {score_rgb:.8f} for rgb case.'


def test_iw_ssim_loss_backprop(x_rand: torch.Tensor, y_rand: torch.Tensor, device: str):
    x_rand.requires_grad_(True)
    loss = InformationWeightedSSIMLoss(data_range=1.)
    score_gray = loss(x_rand.to(device), y_rand.to(device))
    score_gray.backward()
    assert torch.isfinite(x_rand.grad).all(), f'Expected finite gradient values, got {x_rand.grad}.'
