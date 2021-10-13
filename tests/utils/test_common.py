import pytest

from piq.utils import _parse_version as vt


def test_version_tuple_fails_on_empty_string() -> None:
    """ Has to fail because no int values in the empty string """
    with pytest.raises(ValueError):
        vt('')


def test_version_tuple_compares_correctly() -> None:
    assert vt('0.0') < vt('0.0.1')
    assert vt('0.1') < vt('0.2')
    assert vt('1.2.3') < vt('3.2.1')
    assert vt('2.3.1') < vt('2.4.1')
    assert vt('1.0') < vt('2.0')

    # This one indeed should NOT be equal because in python tuples of different lengths cannot be equal
    assert vt('0.0.0') != vt('0.0')

    # But if the length is the same then yes
    assert vt('0.0') == vt('0.0')
