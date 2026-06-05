"""
Test-preview remap colouring.

A remappable material (``Remap1``/``Remap2``/``Remap3``) renders into a reserved
12-shade palette window — the greyscale-looking "company colour" ramps that
OpenRCT2 repaints to the player-chosen colour at draw time.  In a static preview
those windows look wrong, so the generators' ``--test`` path can substitute a
real colour ramp to show what the object looks like once repainted.

This module is preview-only: normal renders leave the remap windows untouched
(OpenRCT2 must repaint them itself).

The ramps in ``REMAP_COLOR_RAMPS`` are the authoritative per-colour shade maps
extracted from OpenRCT2's own ``resources/palettes/map_base/palette_map_*.png``
remap sprites — i.e. exactly the indices OpenRCT2 paints for each shade.  Many
colours share a base ramp with different sub-shade selections (e.g. ``yellow``
and ``dark_yellow`` both live in 46–57), which is why each colour is stored as
an explicit 12-entry ramp rather than a single start index.
"""

__all__ = [
    "REMAP_WINDOWS",
    "REMAP_COLOR_RAMPS",
    "apply_remap_overrides",
    "load_remap_overrides",
]

from typing import Any

import numpy as np

from .types import LoadError

# region -> first palette index of its 12-shade remap window.
# Mirrors the remappable regions in x7_renderer/src/Palette.cpp
# (region 1: 243–254, region 2: 202–213, region 3: 46–57).
REMAP_WINDOWS: dict[int, int] = {
    1: 243,
    2: 202,
    3: 46,
}

_REMAP_SHADES = 12

# OpenRCT2 colour name -> 12 palette indices (darkest..lightest), one per shade
# of a remap window.  Extracted from OpenRCT2 resources/palettes/map_base
# (positions 243–254 of each palette_map_<colour>.png remap sprite).
REMAP_COLOR_RAMPS: dict[str, tuple[int, ...]] = {
    "black": (10, 10, 10, 10, 10, 11, 12, 13, 14, 15, 16, 17),
    "grey": (10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21),
    "white": (13, 14, 15, 16, 17, 18, 19, 20, 21, 21, 21, 21),
    "dark_purple": (118, 118, 118, 119, 119, 120, 121, 122, 122, 123, 124, 124),
    "light_purple": (118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129),
    "bright_purple": (154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165),
    "dark_blue": (130, 130, 130, 131, 131, 132, 133, 134, 134, 135, 136, 136),
    "light_blue": (130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141),
    "icy_blue": (133, 134, 135, 136, 137, 138, 139, 140, 140, 141, 141, 141),
    "teal": (190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201),
    "aquamarine": (191, 193, 195, 196, 197, 198, 199, 200, 200, 201, 201, 201),
    "saturated_green": (94, 94, 94, 95, 95, 96, 97, 98, 98, 99, 100, 100),
    "dark_green": (142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153),
    "moss_green": (70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81),
    "bright_green": (94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105),
    "olive_green": (82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93),
    "dark_olive_green": (22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33),
    "bright_yellow": (48, 49, 50, 51, 52, 53, 54, 55, 56, 56, 57, 57),
    "yellow": (46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57),
    "dark_yellow": (46, 46, 46, 47, 47, 48, 49, 50, 51, 52, 53, 53),
    "light_orange": (178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189),
    "dark_orange": (178, 178, 178, 179, 179, 180, 181, 182, 182, 183, 184, 184),
    "light_brown": (34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45),
    "saturated_brown": (34, 34, 34, 34, 35, 36, 37, 38, 39, 40, 41, 42),
    "dark_brown": (214, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224),
    "salmon_pink": (106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117),
    "bordeaux_red": (58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69),
    "saturated_red": (166, 166, 166, 167, 167, 168, 169, 170, 170, 171, 172, 172),
    "bright_red": (166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177),
    "dark_pink": (202, 202, 202, 203, 203, 204, 205, 206, 206, 207, 208, 208),
    "bright_pink": (202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213),
    "light_pink": (63, 64, 65, 66, 67, 68, 68, 176, 69, 177, 177, 177),
}


def load_remap_overrides(root: dict[str, Any]) -> dict[int, tuple[int, ...]]:
    """Parse the optional ``test_remap_colors`` config block into
    ``{region: ramp}``, where ``ramp`` is the 12-entry shade map for the chosen
    OpenRCT2 colour.

    Returns an empty dict when the block is absent.  Region keys may be given as
    integers or numeric strings so JSON (``"1"``) and YAML (``1``) configs behave
    identically.

    Raises:
        LoadError: if the block is malformed, names an unknown region, or names
            an unknown colour.
    """
    block = root.get("test_remap_colors")
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise LoadError('"test_remap_colors" is not an object')

    out: dict[int, tuple[int, ...]] = {}
    for key, name in block.items():
        try:
            region = int(key)
        except (TypeError, ValueError):
            raise LoadError(f'test_remap_colors key "{key}" is not a region number') from None
        if region not in REMAP_WINDOWS:
            raise LoadError(
                f"test_remap_colors region {region} must be one of {sorted(REMAP_WINDOWS)}"
            )
        if not isinstance(name, str):
            raise LoadError(f"test_remap_colors[{region}] is not a colour name")
        ramp = REMAP_COLOR_RAMPS.get(name)
        if ramp is None:
            raise LoadError(f'Unknown remap colour "{name}" (region {region})')
        out[region] = ramp
    return out


def apply_remap_overrides(
    pixels: np.ndarray, overrides: dict[int, tuple[int, ...]]
) -> np.ndarray:
    """Return a copy of ``pixels`` with each overridden region's remap window
    rewritten to the corresponding shade of the chosen colour ramp.

    A pixel at offset ``k`` within a region's window maps to ``ramp[k]`` — the
    same shade-preserving substitution OpenRCT2 performs when it repaints.
    Pixels outside every overridden window are left untouched.  The original
    array is read throughout, so a target ramp that happens to overlap another
    region's window cannot trigger a second remap.
    """
    if not overrides:
        return pixels
    out = pixels.copy()
    for region, ramp in overrides.items():
        start = REMAP_WINDOWS[region]
        lut = np.asarray(ramp, dtype=np.uint8)
        offsets = pixels.astype(np.int16) - start
        mask = (offsets >= 0) & (offsets < lut.size)
        out[mask] = lut[offsets[mask]]
    return out
