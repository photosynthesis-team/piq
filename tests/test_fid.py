import pytest
import torch

from photosynthesis_metrics import frechet_inception_distance


# ================== Test function: `frechet_inception_distance` ==================
# def test_frechet_inception_distance_fails_for_different_shapes_of_images() -> None:
#     n_items, shape1, shape2 = 10, (3, 64, 64), (3, 128, 128)
#     x, y = torch.rand(n_items, *shape1), torch.rand(100, *shape2)
#     with pytest.raises(AssertionError):
#         frechet_inception_distance(predicted_images=x, target_images=y)


# def test_frechet_inception_distance_works_for_different_number_of_images_in_stack() -> None:
#     n_items1, n_items2, shape = 2, 3, (3, 64, 64)
#     x, y = torch.rand(n_items1, *shape), torch.rand(n_items2, *shape)
#     try:
#         frechet_inception_distance(predicted_images=x, target_images=y)
#     except Exception as e:
#         pytest.fail(f"Unexpected error occurred: {e}")
