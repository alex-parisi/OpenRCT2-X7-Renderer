"""
IndexedImage PNG I/O via Pillow.
"""

__all__ = ["PREVIEW_SIZE", "quantize_to_indexed", "read_png", "write_png"]

from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from .palette import PALETTE_RGB, TRANSPARENT_INDEX
from .types import IndexedImage


def read_png(path: Path | str) -> IndexedImage:
    """Read a paletted PNG into an IndexedImage.

    ``x_offset`` and ``y_offset`` are always set to 0 because PNG has no field
    for OpenRCT2 sprite draw offsets; they are lost in a ``write_png`` /
    ``read_png`` round-trip.
    """
    with PILImage.open(path) as img:
        if img.mode != "P":
            raise ValueError(f"PNG {path} is not paletted (mode={img.mode})")
        pixels = np.array(img, dtype=np.uint8)
    height, width = pixels.shape
    return IndexedImage(
        width=width,
        height=height,
        x_offset=0,
        y_offset=0,
        pixels=pixels,
    )


# Vanilla ride previews size:
PREVIEW_SIZE = 112

# Palette indices that are safe to quantize arbitrary imagery into.
# Skip the transparent/special low entries (0-9) and the remap/cycling high entries
# (237+, which include the remap1/remap2 regions and animated colours)
_QUANTIZE_FIRST = 10
_QUANTIZE_LAST = 236


def quantize_to_indexed(path: Path | str, *, size: int = PREVIEW_SIZE) -> IndexedImage:
    """Read any image file, resize to fit ``size`` x ``size``, and quantize it
    to the RCT2 palette.

    Unlike :func:`read_png` (which requires an already-paletted RCT2 PNG), this
    accepts arbitrary formats and colour modes. The source is resized in RGB
    with Lanczos resampling, then Floyd-Steinberg dithered onto the non-remap
    region of the palette. A source alpha channel maps to the transparent
    palette index.
    """
    with PILImage.open(path) as src:
        rgba = src.convert("RGBA")
    # thumbnail() only shrinks; also upscale so small sources fill the target box.
    rgba.thumbnail((size, size), PILImage.Resampling.LANCZOS)
    if max(rgba.width, rgba.height) < size:
        scale = size / max(rgba.width, rgba.height)
        rgba = rgba.resize(
            (round(rgba.width * scale), round(rgba.height * scale)),
            PILImage.Resampling.LANCZOS,
        )

    # Build a full 256-entry palette by tiling the candidate colours
    candidate_indices = np.arange(_QUANTIZE_FIRST, _QUANTIZE_LAST + 1, dtype=np.uint8)
    candidate_rgb = PALETTE_RGB[candidate_indices]
    count = len(candidate_indices)
    tiled = np.arange(256) % count
    pal_rgb = candidate_rgb[tiled]
    local_to_real = candidate_indices[tiled]

    pal_img = PILImage.new("P", (1, 1))
    pal_img.putpalette(pal_rgb.flatten().tolist())

    quantized = rgba.convert("RGB").quantize(palette=pal_img, dither=PILImage.Dither.FLOYDSTEINBERG)
    pixels = local_to_real[np.array(quantized, dtype=np.uint8)]

    # Honour source transparency
    alpha = np.array(rgba)[:, :, 3]
    pixels[alpha < 128] = TRANSPARENT_INDEX

    height, width = pixels.shape
    return IndexedImage(width=width, height=height, x_offset=0, y_offset=0, pixels=pixels)


def write_png(image: IndexedImage, path: Path | str) -> None:
    """Write IndexedImage as a paletted PNG with transparent index 0."""
    img = PILImage.fromarray(image.pixels, mode="P")
    img.putpalette(PALETTE_RGB.tobytes())
    img.save(path, format="PNG", optimize=False, transparency=TRANSPARENT_INDEX)
