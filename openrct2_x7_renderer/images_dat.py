"""
OpenRCT2 `images.dat` (G1) blob writer
"""

import struct
from pathlib import Path

from .types import IndexedImage

# OpenRCT2 G1 image flag bits.
G1_FLAG_BMP = 0x0001


def write_images_dat(images: list[IndexedImage], out_path: Path) -> None:
    """
    Write a sequence of IndexedImages as an OpenRCT2 `images.dat` (G1) blob.

    Format (matches the vanilla parkobj's `images.dat`):
      - Header (8 bytes): u32 num_entries, u32 total_pixel_data_size.
      - num_entries * 16-byte G1 elements:
          u32 offset (into the pixel data section),
          u16 width, u16 height, i16 x_offset, i16 y_offset,
          u16 flags, u16 zoom (we always write 0).
      - Concatenated pixel data: each image is width*height bytes of
        palette indices, with index 0 acting as transparent.

    The matching `object.json` `images` entry is the single string
    `"$LGX:images.dat[0..N-1]"`.
    """
    num = len(images)
    offsets: list[int] = []
    chunks: list[bytes] = []
    cur = 0
    for i, img in enumerate(images):
        if not (0 <= img.width <= 65535 and 0 <= img.height <= 65535):
            raise ValueError(
                f"Image {i}: width ({img.width}) and height ({img.height}) "
                f"must fit in an unsigned 16-bit integer"
            )
        if not (-32768 <= img.x_offset <= 32767 and -32768 <= img.y_offset <= 32767):
            raise ValueError(
                f"Image {i}: x_offset ({img.x_offset}) and y_offset ({img.y_offset}) "
                f"must fit in a signed 16-bit integer"
            )
        pixels = img.pixels.tobytes()  # uint8 (H, W) row-major
        offsets.append(cur)
        chunks.append(pixels)
        cur += len(pixels)
    total_pixel_size = cur

    elements = bytearray()
    for img, offset in zip(images, offsets, strict=True):
        elements += struct.pack(
            "<IHHhhHH",
            offset,
            int(img.width),
            int(img.height),
            int(img.x_offset),
            int(img.y_offset),
            G1_FLAG_BMP,
            0,
        )

    with Path(out_path).open("wb") as f:
        f.write(struct.pack("<II", num, total_pixel_size))
        f.write(bytes(elements))
        f.writelines(chunks)
