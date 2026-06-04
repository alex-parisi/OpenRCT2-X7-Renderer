"""
RCT2 256-color palette — the C++ extension (``_x7_renderer``) is the single
source of truth.  ``srgb2linear`` is also exported for modules that perform
texture loading independently of the renderer.
"""

__all__ = ["PALETTE_RGB", "TRANSPARENT_INDEX", "srgb2linear"]

import numpy as np
from ._x7_renderer import palette_rgb as _native_palette_rgb

TRANSPARENT_INDEX = 0
PALETTE_RGB: np.ndarray = _native_palette_rgb()


def srgb2linear(x: np.ndarray) -> np.ndarray:
    """Convert sRGB values in [0, 1] to linear light values using the IEC 61966-2-1 curve."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    lo = x <= 0.04045
    out[lo] = x[lo] / 12.92
    out[~lo] = np.power((x[~lo] + 0.055) / 1.055, 2.4)
    return out
