import pytest
import torch
import subprocess
import sys
import builtins

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
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package])


def prepare_test(scipy_version='1.3.3', gudhi_version='3.2') -> None:
    try:
        import scipy  # noqa: F401
    except ImportError:
        install('scipy' + '==' + scipy_version)

    try:
        import gudhi  # noqa: F401
    except ImportError:
        install('gudhi' + '==' + gudhi_version)


@pytest.fixture
def hide_available_pkg(monkeypatch):
    import_orig = builtins.__import__

    def mocked_import(name, *args, **kwargs):
        if name in ['scipy', 'gudhi']:
            raise ImportError()

        return import_orig(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', mocked_import)


# ================== Test class: `GS` ==================
def test_initialization() -> None:
    prepare_test()
    try:
        GS()
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")


@pytest.mark.usefixtures('hide_available_pkg')
def test_fails_if_libs_not_installed(features_y_normal, features_x_normal) -> None:
    with pytest.raises(ImportError):
        metric = GS(num_iters=10, sample_size=8)
        metric(features_y_normal, features_x_normal)


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
