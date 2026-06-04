"""Round-trip test for the OpenRCT2 images.dat (G1 blob) writer."""

import struct

import numpy as np
from openrct2_x7_renderer.images_dat import G1_FLAG_BMP, write_images_dat
from openrct2_x7_renderer.types import IndexedImage


def _img(w, h, x_offset, y_offset, fill):
    return IndexedImage(
        width=w,
        height=h,
        x_offset=x_offset,
        y_offset=y_offset,
        pixels=np.full((h, w), fill, dtype=np.uint8),
    )


def _parse_images_dat(data: bytes):
    num, total = struct.unpack_from("<II", data, 0)
    elements = []
    elem_base = 8
    for i in range(num):
        off, w, h, xo, yo, flags, zoom = struct.unpack_from("<IhhhhHH", data, elem_base + i * 16)
        elements.append(dict(offset=off, w=w, h=h, x=xo, y=yo, flags=flags, zoom=zoom))
    pixel_base = elem_base + num * 16
    return num, total, elements, data[pixel_base:]


def test_images_dat_roundtrip(tmp_path):
    images = [
        _img(2, 3, -1, -2, 7),
        _img(4, 1, 5, 6, 9),
        _img(1, 1, 0, 0, 255),
    ]
    out = tmp_path / "images.dat"
    write_images_dat(images, out)

    num, total, elements, pixels = _parse_images_dat(out.read_bytes())

    assert num == 3
    assert total == sum(i.width * i.height for i in images)
    assert len(pixels) == total

    cursor = 0
    for img, el in zip(images, elements, strict=False):
        assert el["w"] == img.width
        assert el["h"] == img.height
        assert el["x"] == img.x_offset
        assert el["y"] == img.y_offset
        assert el["flags"] == G1_FLAG_BMP
        assert el["zoom"] == 0
        assert el["offset"] == cursor
        chunk = pixels[cursor : cursor + img.width * img.height]
        assert np.array_equal(
            np.frombuffer(chunk, dtype=np.uint8).reshape(img.height, img.width),
            img.pixels,
        )
        cursor += img.width * img.height


def test_images_dat_offsets_are_monotonic(tmp_path):
    images = [_img(w, w, 0, 0, w) for w in (1, 2, 3, 4)]
    out = tmp_path / "images.dat"
    write_images_dat(images, out)
    _num, _total, elements, _pixels = _parse_images_dat(out.read_bytes())
    offsets = [e["offset"] for e in elements]
    assert offsets == sorted(offsets)
    assert offsets[0] == 0
