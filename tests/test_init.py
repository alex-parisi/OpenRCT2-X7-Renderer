"""Tests for the openrct2_x7_renderer package __init__ module."""

import importlib
import importlib.metadata
from importlib.metadata import PackageNotFoundError
from unittest.mock import patch


def test_version_is_a_string():
    import openrct2_x7_renderer

    assert isinstance(openrct2_x7_renderer.__version__, str)
    assert len(openrct2_x7_renderer.__version__) > 0


def test_version_fallback_when_package_not_found():
    """The except-PackageNotFoundError branch in __init__ sets a dev fallback."""
    import openrct2_x7_renderer

    with patch("importlib.metadata.version", side_effect=PackageNotFoundError):
        importlib.reload(openrct2_x7_renderer)

    assert openrct2_x7_renderer.__version__ == "0.0.0.dev0"

    # Restore the real version so subsequent tests see the installed value.
    importlib.reload(openrct2_x7_renderer)
