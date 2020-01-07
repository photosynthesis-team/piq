import unittest
import torch


from metrics import ssim, SSIMLoss


class SSIMTest(unittest.TestCase):
    def setUp(self) -> None:
        self.prediction = torch.rand(3, 3, 256, 256)
        self.target = torch.rand(3, 3, 256, 256)

    def tearDown(self) -> None:
        del self.prediction, self.target

    def test_symmetry(self) -> None:
        measure = ssim(self.prediction, self.target, data_range=1.)
        reverse_measure = ssim(self.target, self.prediction, data_range=1.)
        self.assertEqual(measure, reverse_measure,
                         msg=f'Expect: SSIM(a, b) == SSIM(b, a), got {measure} != {reverse_measure}')

    @unittest.skipIf(not torch.cuda.is_available(), 'No need to run test on GPU if there is no GPU.')
    def test_symmetry_cuda(self) -> None:
        self.prediction.cuda()
        self.target.cuda()
        self.test_symmetry()

    def test_measure_is_zero_for_equal_tensors(self) -> None:
        prediction = self.target.clone()
        measure = ssim(prediction, self.target, data_range=1.)
        measure = measure - 1.
        self.assertEqual(measure.sum(), 0, msg=f'If equal tensors are passed SSIM must be equal to 0, got {measure}')

    @unittest.skipIf(not torch.cuda.is_available(), 'No need to run test on GPU if there is no GPU.')
    def test_measure_is_zero_for_equal_tensors_cuda(self) -> None:
        self.target.cuda()
        self.test_measure_is_zero_for_equal_tensors()

    def test_measure_is_less_or_equal_to_one(self) -> None:
        # Create two maximally different tensors.
        ones = torch.ones((3, 3, 256, 256))
        zeros = torch.zeros((3, 3, 256, 256))
        measure = ssim(ones, zeros, data_range=1.)
        self.assertLessEqual(measure, 1, msg=f'SSIM must be <= 1, got {measure}')

    @unittest.skipIf(not torch.cuda.is_available(), 'No need to run test on GPU if there is no GPU.')
    def test_measure_is_less_or_equal_to_one_cuda(self) -> None:
        ones = torch.ones((3, 3, 256, 256)).cuda()
        zeros = torch.zeros((3, 3, 256, 256)).cuda()
        measure = ssim(ones, zeros, data_range=1.)
        self.assertLessEqual(measure, 1, msg=f'SSIM must be <= 1, got {measure}')

    def test_raises_if_tensors_have_different_shapes(self) -> None:
        wrong_shape_prediction = torch.rand(3, 2, 64, 64)
        with self.assertRaises(AssertionError):
            ssim(wrong_shape_prediction, self.target)

    def test_raises_if_tensors_have_different_types(self) -> None:
        wrong_type_prediction = list(range(10))
        with self.assertRaises(AssertionError):
            ssim(wrong_type_prediction, self.target)

    def test_raises_if_wrong_kernel_size_is_passed(self) -> None:
        wrong_kernel_sizes = list(range(0, 50, 2))
        for kernel_size in wrong_kernel_sizes:
            with self.assertRaises(AssertionError):
                ssim(self.prediction, self.target, kernel_size=kernel_size)


class SSIMLossTest(unittest.TestCase):
    def setUp(self) -> None:
        self.prediction = torch.rand(3, 3, 256, 256)
        self.target = torch.rand(3, 3, 256, 256)

    def tearDown(self) -> None:
        del self.prediction, self.target

    def test_symmetry(self) -> None:
        loss = SSIMLoss()
        loss_value = loss(self.prediction, self.target, data_range=1.)
        reverse_loss_value = loss(self.target, self.prediction, data_range=1.)
        self.assertEqual(loss_value, reverse_loss_value,
                         msg=f'Expect: SSIM(a, b) == SSIM(b, a), got {loss_value} != {reverse_loss_value}')

    @unittest.skipIf(not torch.cuda.is_available(), 'No need to run test on GPU if there is no GPU.')
    def test_symmetry_cuda(self) -> None:
        self.prediction.cuda()
        self.target.cuda()
        self.test_symmetry()

    def test_equality(self) -> None:
        prediction = self.target.clone()
        measure = SSIMLoss()(prediction, self.target, data_range=1.)
        measure = measure - 1.
        self.assertEqual(measure.sum(), 0,
                         msg=f'If equal tensors are passed SSIM loss must be equal to 0, got {measure}')

    @unittest.skipIf(not torch.cuda.is_available(), 'No need to run test on GPU if there is no GPU.')
    def test_equality_cuda(self) -> None:
        self.target.cuda()
        self.test_equality()

    def test_measure_is_less_or_equal_to_one(self) -> None:
        # Create two maximally different tensors.
        ones = torch.ones((3, 3, 256, 256))
        zeros = torch.zeros((3, 3, 256, 256))
        measure = SSIMLoss()(ones, zeros, data_range=1.)
        self.assertLessEqual(measure, 1, msg=f'SSIM loss must be <= 1, got {measure}')

    @unittest.skipIf(not torch.cuda.is_available(), 'No need to run test on GPU if there is no GPU.')
    def test_measure_is_less_or_equal_to_one_cuda(self) -> None:
        ones = torch.ones((3, 3, 256, 256)).cuda()
        zeros = torch.zeros((3, 3, 256, 256)).cuda()
        measure = SSIMLoss()(ones, zeros, data_range=1.)
        self.assertLessEqual(measure, 1, msg=f'SSIM loss must be <= 1, got {measure}')
