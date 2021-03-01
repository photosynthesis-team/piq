import pytest
import torch
import subprocess
import sys
import warnings

from piq import GS


@pytest.fixture(scope='module')
def features_y_normal() -> torch.Tensor:
    return torch.rand(1000, 20)


@pytest.fixture(scope='module')
def features_x_normal() -> torch.Tensor:
    return torch.rand(1000, 20)


@pytest.fixture(scope='module')
def features_x_beta() -> torch.Tensor:
    m = torch.distributions.Beta(torch.FloatTensor([2]), torch.FloatTensor([2]))
    return m.sample([1000, 20]).squeeze()


def install(package: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def uninstall(package) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", package])


def prepare_test() -> None:
    try:
        import scipy
    except ImportError:
        install('scipy')

    try:
        import gudhi
    except ImportError:
        install('scipy')


# ================== Test class: `GS` ==================
def test_initialization() -> None:
    prepare_test()
    try:
        GS()
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


def test_fails_is_libs_not_install(features_y_normal, features_x_normal) -> None:
    reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
    installed_packages = [r.decode().split('==')[0] for r in reqs.split()]
    if 'scipy' in installed_packages:
        uninstall('scipy')

    if 'gudhi' in installed_packages:
        uninstall('gudhi')

    with pytest.raises(ImportError):
        metric = GS(num_iters=10, sample_size=8)
        metric(features_y_normal, features_x_normal)


def test_warns_if_lowe_versions(features_y_normal, features_x_normal) -> None:
    prepare_test()

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered
        warnings.simplefilter("always")

        # Trigger a warnings
        try:
            metric = GS(num_iters=10, sample_size=8)
            metric(features_y_normal, features_x_normal)
        except Exception as e:
            pytest.fail(f"Unexpected error occurred: {e}")

        # Verify some things
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "deprecated" in str(w[-1].message)


@pytest.mark.skip(reason="Randomnly fails, fix in separate PR")
def test_similar_for_same_distribution(features_y_normal, features_x_normal) -> None:
    prepare_test()
    metric = GS(sample_size=1000, num_iters=100, i_max=1000, num_workers=4)
    diff = metric(features_x_normal, features_y_normal)
    assert diff <= 2.0, \
        f'For same distributions GS should be small, got {diff}'


@pytest.mark.skip(reason="Randomnly fails, fix in separate PR")
def test_differs_for_not_simular_distributions(features_x_beta, features_y_normal) -> None:
    prepare_test()
    metric = GS(sample_size=1000, num_iters=100, i_max=1000, num_workers=4)
    diff = metric(features_x_beta, features_y_normal)
    assert diff >= 5.0, \
        f'For different distributions GS diff should be big, got {diff}'
