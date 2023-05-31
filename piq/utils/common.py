import torch
import re
import os
import warnings

from typing import Tuple, List, Optional, Union, Dict, Any
from urllib.request import urlopen

SEMVER_VERSION_PATTERN = re.compile(
    r"""
        ^
        (?P<major>0|[1-9]\d*)
        \.
        (?P<minor>0|[1-9]\d*)
        \.
        (?P<patch>0|[1-9]\d*)
        (?:-(?P<prerelease>
            (?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)
            (?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*
        ))?
        (?:\+(?P<build>
            [0-9a-zA-Z-]+
            (?:\.[0-9a-zA-Z-]+)*
        ))?
        $
    """,
    re.VERBOSE,
)


PEP_440_VERSION_PATTERN = r"""
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\.]?
            (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))
            [-_\.]?
            (?P<pre_n>[0-9]+)?
        )?
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\.]?
                (?P<post_l>post|rev|r)
                [-_\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
        (?P<dev>                                          # dev release
            [-_\.]?
            (?P<dev_l>dev)
            [-_\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
"""


def _validate_input(
        tensors: List[torch.Tensor],
        dim_range: Tuple[int, int] = (0, -1),
        data_range: Tuple[float, float] = (0., -1.),
        size_range: Optional[Tuple[int, int]] = None,
        check_for_channels_first: bool = False
) -> None:
    r"""Check that input(-s)  satisfies the requirements
    Args:
        tensors: Tensors to check
        dim_range: Allowed number of dimensions. (min, max)
        data_range: Allowed range of values in tensors. (min, max)
        size_range: Dimensions to include in size comparison. (start_dim, end_dim + 1)
    """

    if not __debug__:
        return

    x = tensors[0]

    for t in tensors:
        assert torch.is_tensor(t), f'Expected torch.Tensor, got {type(t)}'
        assert t.device == x.device, f'Expected tensors to be on {x.device}, got {t.device}'

        if size_range is None:
            assert t.size() == x.size(), f'Expected tensors with same size, got {t.size()} and {x.size()}'
        else:
            assert t.size()[size_range[0]: size_range[1]] == x.size()[size_range[0]: size_range[1]], \
                f'Expected tensors with same size at given dimensions, got {t.size()} and {x.size()}'

        if dim_range[0] == dim_range[1]:
            assert t.dim() == dim_range[0], f'Expected number of dimensions to be {dim_range[0]}, got {t.dim()}'
        elif dim_range[0] < dim_range[1]:
            assert dim_range[0] <= t.dim() <= dim_range[1], \
                f'Expected number of dimensions to be between {dim_range[0]} and {dim_range[1]}, got {t.dim()}'

        if data_range[0] < data_range[1]:
            assert data_range[0] <= t.min(), \
                f'Expected values to be greater or equal to {data_range[0]}, got {t.min()}'
            assert t.max() <= data_range[1], \
                f'Expected values to be lower or equal to {data_range[1]}, got {t.max()}'
            
        if check_for_channels_first:
            channels_last = t.shape[-1] in {1, 2, 3}
            assert not channels_last, "Expected tensor to have channels first format, but got channels last. \
                Please permute channels (e.g. t.permute(0, 3, 1, 2) for 4D tensors) and rerun."


def _reduce(x: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    r"""Reduce input in batch dimension if needed.

    Args:
        x: Tensor with shape (N, *).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``
    """
    if reduction == 'none':
        return x
    elif reduction == 'mean':
        return x.mean(dim=0)
    elif reduction == 'sum':
        return x.sum(dim=0)
    else:
        raise ValueError("Unknown reduction. Expected one of {'none', 'mean', 'sum'}")


def _parse_version(version: Union[str, bytes]) -> Tuple[int, ...]:
    """ Parses valid Python versions according to Semver and PEP 440 specifications.
    For more on Semver check: https://semver.org/
    For more on PEP 440 check: https://www.python.org/dev/peps/pep-0440/.

    Implementation is inspired by:
    - https://github.com/python-semver
    - https://github.com/pypa/packaging

    Args:
        version: unparsed information about the library of interest.

    Returns:
        parsed information about the library of interest.
    """
    if isinstance(version, bytes):
        version = version.decode("UTF-8")
    elif not isinstance(version, str) and not isinstance(version, bytes):
        raise TypeError(f"not expecting type {type(version)}")

    # Semver processing
    match = SEMVER_VERSION_PATTERN.match(version)
    if match:
        matched_version_parts: Dict[str, Any] = match.groupdict()
        release = tuple([int(matched_version_parts[k]) for k in ['major', 'minor', 'patch']])
        return release

    # PEP 440 processing
    regex = re.compile(r"^\s*" + PEP_440_VERSION_PATTERN + r"\s*$", re.VERBOSE | re.IGNORECASE)
    match = regex.search(version)

    if match is None:
        warnings.warn(f"{version} is not a valid SemVer or PEP 440 string")
        return tuple()

    release = tuple(int(i) for i in match.group("release").split("."))
    return release


def download_tensor(url: str, root: str, map_location: str = 'cpu') -> torch.Tensor:
    r"""Downloads torch tensor and caches it. If already downloaded - loads from cache.

    Args:
        url: Web or file system path.
        root: Absolute or relative path of the cache folder.

    Returns:
        Loaded torch tesnor.
    """
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    download_target = os.path.join(root, filename)
    if os.path.isfile(download_target):
        return torch.load(download_target, map_location=map_location)
    
    with urlopen(url) as source, open(download_target, "wb") as output:
        while True:
            buffer = source.read(8192)
            if not buffer:
                break

            output.write(buffer)

    return torch.load(download_target, map_location=map_location)


def is_sha256_hash(string: str) -> Optional[re.Match]:
    """ Checks whether the provided sting is a valid SHA256 hash. """
    pattern = re.compile("^[a-fA-F0-9]{64}$")
    return pattern.match(string)
