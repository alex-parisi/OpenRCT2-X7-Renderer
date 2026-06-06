"""
Rendering primitives shared by the vehicle and scenery generators.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


class LoadError(Exception):
    """Raised when a config file or asset is malformed or cannot be loaded."""


@dataclass(frozen=True, slots=True, eq=False)
class MeshFrame:
    """One animation frame of a mesh placement within a Model.

    Attributes:
        mesh_index: Index into the meshes list, or ``-1`` if this slot is unused.
        position: ``(3,)`` translation in OBJ units applied after rotation.
        orientation: ``(3,)`` Euler angles in degrees ``(angle_y, angle_z, angle_x)`` applied
        as ``rotate_y @ rotate_z @ rotate_x``.
    """

    mesh_index: int = -1
    position: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    orientation: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3, dtype=np.float64))


@dataclass
class Model:
    """A scene object composed of one or more animated mesh placements.

    Attributes:
        meshes: List of placements; each placement is a list of MeshFrames (one per
            animation frame), allowing the mesh index and transform to vary per frame.
    """

    meshes: list[list[MeshFrame]] = field(default_factory=list)


@dataclass(frozen=True, slots=True, eq=False)
class IndexedImage:
    """8-bit palette image. Pixel value 0 is transparent.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        x_offset: Horizontal draw offset relative to the sprite origin.
        y_offset: Vertical draw offset relative to the sprite origin.
        pixels: uint8 ``(height, width)`` array of palette indices; 0 is transparent.
    """

    width: int
    height: int
    x_offset: int
    y_offset: int
    pixels: NDArray[np.uint8]  # (height, width)

    @classmethod
    def blank(cls, width: int, height: int, x_offset: int = 0, y_offset: int = 0) -> "IndexedImage":
        """Return an all-transparent IndexedImage of the given dimensions."""
        return cls(
            width=width,
            height=height,
            x_offset=x_offset,
            y_offset=y_offset,
            pixels=np.zeros((height, width), dtype=np.uint8),
        )


@dataclass(frozen=True, slots=True, eq=False)
class Light:
    """A single light source in the render rig.

    Attributes:
        type: Light type constant — one of ``LightType.HEMI``, ``LightType.DIFFUSE``,
            or ``LightType.SPECULAR`` from ``constants.py``.
        shadow: True if the light casts shadows.
        direction: ``(3,)`` unit vector pointing toward the light source.
        intensity: Light strength multiplier.
    """

    type: int
    shadow: bool
    direction: NDArray[np.float64]
    intensity: float
