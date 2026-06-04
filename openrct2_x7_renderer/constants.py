"""
Shared rendering constants.
Ported from X7's rendering engine
https://github.com/X123M3-256/RCTGen

The MATERIAL_*, MESH_*, and LIGHT_* constants are defined in C++
(``x7_renderer/src/Mesh.hpp`` and ``x7_renderer/src/Renderer.hpp``) and
imported here from the compiled extension so there is a single source of truth.
"""

__all__ = [
    "LightType",
    "MaterialFlag",
    "MeshFlag",
    "TILE_SIZE",
]

from enum import IntEnum, IntFlag

from ._x7_renderer import (
    LIGHT_DIFFUSE as _LIGHT_DIFFUSE,
    LIGHT_HEMI as _LIGHT_HEMI,
    LIGHT_SPECULAR as _LIGHT_SPECULAR,
    MATERIAL_BACKGROUND_AA as _MATERIAL_BACKGROUND_AA,
    MATERIAL_BACKGROUND_AA_DARK as _MATERIAL_BACKGROUND_AA_DARK,
    MATERIAL_HAS_TEXTURE as _MATERIAL_HAS_TEXTURE,
    MATERIAL_IS_FLAT_SHADED as _MATERIAL_IS_FLAT_SHADED,
    MATERIAL_IS_MASK as _MATERIAL_IS_MASK,
    MATERIAL_IS_REMAPPABLE as _MATERIAL_IS_REMAPPABLE,
    MATERIAL_IS_VISIBLE_MASK as _MATERIAL_IS_VISIBLE_MASK,
    MATERIAL_NO_AO as _MATERIAL_NO_AO,
    MATERIAL_NO_BLEED as _MATERIAL_NO_BLEED,
    MESH_GHOST as _MESH_GHOST,
    MESH_MASK as _MESH_MASK,
)


class MaterialFlag(IntFlag):
    """Bitmask flags for material rendering properties.

    Most flags are set automatically by :func:`~openrct2_x7_renderer.mesh.load_mesh`
    when it finds matching substrings in OBJ material names (e.g. ``Remap1``,
    ``NoAO``, ``DarkEdge``).  ``IS_FLAT_SHADED`` is reserved for future use
    and has no effect on rendering currently; it also has no corresponding
    material-name keyword.
    """

    HAS_TEXTURE = _MATERIAL_HAS_TEXTURE
    IS_REMAPPABLE = _MATERIAL_IS_REMAPPABLE
    IS_MASK = _MATERIAL_IS_MASK
    NO_AO = _MATERIAL_NO_AO
    BACKGROUND_AA = _MATERIAL_BACKGROUND_AA
    BACKGROUND_AA_DARK = _MATERIAL_BACKGROUND_AA_DARK
    IS_VISIBLE_MASK = _MATERIAL_IS_VISIBLE_MASK
    NO_BLEED = _MATERIAL_NO_BLEED
    IS_FLAT_SHADED = _MATERIAL_IS_FLAT_SHADED  # reserved; not currently implemented by the renderer


class MeshFlag(IntFlag):
    """Bitmask flags for mesh rendering behaviour."""

    MASK = _MESH_MASK
    GHOST = _MESH_GHOST


class LightType(IntEnum):
    """Light source type constants."""

    HEMI = _LIGHT_HEMI
    DIFFUSE = _LIGHT_DIFFUSE
    SPECULAR = _LIGHT_SPECULAR


TILE_SIZE = 3.3
