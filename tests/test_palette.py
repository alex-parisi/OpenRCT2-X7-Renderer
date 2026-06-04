"""Tests for the RCT2 palette tables."""

import numpy as np
from openrct2_x7_renderer.palette import (
    PALETTE_RGB,
    TRANSPARENT_INDEX,
    srgb2linear,
)


def test_palette_shape_and_dtype():
    assert PALETTE_RGB.shape == (256, 3)
    assert PALETTE_RGB.dtype == np.uint8


def test_transparent_index_is_zero():
    assert TRANSPARENT_INDEX == 0


def test_srgb2linear_black_and_white():
    result = srgb2linear(np.array([0.0, 1.0]))
    assert np.isclose(result[0], 0.0)
    assert np.isclose(result[1], 1.0)


def test_srgb2linear_low_value_uses_linear_segment():
    # Values <= 0.04045 map through the linear segment x/12.92.
    x = np.array([0.0, 0.04045])
    result = srgb2linear(x)
    assert np.allclose(result, x / 12.92)


def test_srgb2linear_high_value_uses_gamma_segment():
    # Values > 0.04045 use the power curve.
    x = np.array([0.5, 1.0])
    result = srgb2linear(x)
    expected = np.power((x + 0.055) / 1.055, 2.4)
    assert np.allclose(result, expected)


def test_srgb2linear_mixed_input():
    # Verify both branches are applied correctly in a single call.
    x = np.array([0.0, 0.04045, 0.5])
    result = srgb2linear(x)
    assert result[0] == 0.0
    assert np.isclose(result[1], 0.04045 / 12.92)
    assert np.isclose(result[2], ((0.5 + 0.055) / 1.055) ** 2.4)


def test_palette_rgb_derives_from_extension():
    """PALETTE_RGB comes from the C++ extension (palette_rgb()), not a Python table."""
    from openrct2_x7_renderer import _x7_renderer as x7

    assert np.array_equal(PALETTE_RGB, x7.palette_rgb())
