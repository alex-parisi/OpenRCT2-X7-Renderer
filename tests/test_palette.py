"""Tests for the RCT2 palette tables."""

import numpy as np
from openrct2_x7_renderer.palette import (
    PALETTE_RGB,
    TRANSPARENT_INDEX,
)


def test_palette_shape_and_dtype():
    assert PALETTE_RGB.shape == (256, 3)
    assert PALETTE_RGB.dtype == np.uint8


def test_transparent_index_is_zero():
    assert TRANSPARENT_INDEX == 0
