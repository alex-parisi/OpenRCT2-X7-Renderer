"""Tests for IndexedImage PNG I/O and palette quantization."""

import numpy as np
import pytest
from openrct2_x7_renderer.image import (
    quantize_to_indexed,
    read_png,
    write_png,
)
from openrct2_x7_renderer.palette import PALETTE_RGB, TRANSPARENT_INDEX
from openrct2_x7_renderer.types import IndexedImage
from PIL import Image as PILImage


def _indexed(pixels):
    arr = np.array(pixels, dtype=np.uint8)
    h, w = arr.shape
    return IndexedImage(width=w, height=h, x_offset=0, y_offset=0, pixels=arr)


def test_png_write_read_roundtrip(tmp_path):
    img = _indexed([[0, 1, 2], [10, 100, 200]])
    path = tmp_path / "img.png"
    write_png(img, path)
    back = read_png(path)
    assert back.width == 3
    assert back.height == 2
    assert np.array_equal(back.pixels, img.pixels)


def test_png_is_paletted_with_rct2_palette(tmp_path):
    path = tmp_path / "img.png"
    write_png(_indexed([[5, 6], [7, 8]]), path)
    with PILImage.open(path) as pil:
        assert pil.mode == "P"
        pal = np.array(pil.getpalette(), dtype=np.uint8).reshape(-1, 3)
        # The written PLTE must match the RCT2 image palette for the indices used.
        assert np.array_equal(pal[:9], PALETTE_RGB[:9])
        assert pil.info["transparency"] == TRANSPARENT_INDEX


def test_read_png_rejects_non_paletted(tmp_path):
    path = tmp_path / "rgb.png"
    PILImage.new("RGB", (2, 2), (1, 2, 3)).save(path)
    with pytest.raises(ValueError, match="not paletted"):
        read_png(path)


def test_quantize_resizes_within_bounds(tmp_path):
    path = tmp_path / "big.png"
    PILImage.new("RGB", (400, 200), (120, 60, 30)).save(path)
    out = quantize_to_indexed(path, size=112)
    assert out.width <= 112
    assert out.height <= 112
    # A solid colour must not land in the transparent/remap-reserved ranges.
    assert out.pixels.min() >= 10
    assert out.pixels.max() <= 236


def test_quantize_maps_alpha_to_transparent(tmp_path):
    path = tmp_path / "alpha.png"
    rgba = PILImage.new("RGBA", (16, 16), (200, 50, 50, 0))  # fully transparent
    rgba.save(path)
    out = quantize_to_indexed(path, size=16)
    assert np.all(out.pixels == TRANSPARENT_INDEX)


def test_quantize_solid_opaque_has_no_transparent_pixels(tmp_path):
    path = tmp_path / "solid.png"
    PILImage.new("RGBA", (16, 16), (90, 140, 200, 255)).save(path)
    out = quantize_to_indexed(path, size=16)
    assert not np.any(out.pixels == TRANSPARENT_INDEX)
