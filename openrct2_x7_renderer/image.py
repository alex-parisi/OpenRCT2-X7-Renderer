"""
IndexedImage PNG I/O via Pillow.

Ports the PNG read/write portion of src/iso-render/Image.cpp.
"""

from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from .palette import PALETTE_RGB, TRANSPARENT_INDEX
from .types import IndexedImage


def read_png(path: Path | str) -> IndexedImage:
    """Read a paletted PNG into an IndexedImage. Mirrors image_read_png."""
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


# Object-picker previews are square thumbnails; this is the size the vanilla
# ride previews (and the bundled example) use.
PREVIEW_SIZE = 112

# Palette indices that are safe to quantize arbitrary imagery into: skip the
# transparent/special low entries (0-9) and the remap/cycling high entries
# (237+, which include the remap1/remap2 regions and animated colours). Static
# preview images should never land in those.
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
    rgba.thumbnail((size, size), PILImage.Resampling.LANCZOS)

    # Build a full 256-entry palette by tiling the candidate colours, so PIL's
    # quantizer can never pick an unused (black) slot; map each local index
    # back to the real RCT2 palette index it stands for.
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

    # Honour source transparency: anything mostly-transparent becomes index 0.
    alpha = np.array(rgba)[:, :, 3]
    pixels[alpha < 128] = TRANSPARENT_INDEX

    height, width = pixels.shape
    return IndexedImage(width=width, height=height, x_offset=0, y_offset=0, pixels=pixels)


def write_png(image: IndexedImage, path: Path | str) -> None:
    """Write IndexedImage as a paletted PNG with transparent index 0."""
    img = PILImage.fromarray(image.pixels, mode="P")
    img.putpalette(PALETTE_RGB.tobytes())
    img.info["transparency"] = TRANSPARENT_INDEX
    img.save(path, format="PNG", optimize=False)
