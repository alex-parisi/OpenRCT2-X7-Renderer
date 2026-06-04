"""
Rendering primitives shared by the vehicle and scenery generators.
"""

from dataclasses import dataclass, field

import numpy as np

MAX_FRAMES = 4


@dataclass
class MeshFrame:
    mesh_index: int = -1
    position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    orientation: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))


@dataclass
class Model:
    meshes: list[list[MeshFrame]] = field(default_factory=list)


@dataclass
class IndexedImage:
    """8-bit palette image. Pixel value 0 is transparent."""

    width: int
    height: int
    x_offset: int
    y_offset: int
    pixels: np.ndarray  # uint8 (height, width)

    @classmethod
    def blank(cls, width: int, height: int, x_offset: int = 0, y_offset: int = 0) -> "IndexedImage":
        return cls(
            width=width,
            height=height,
            x_offset=x_offset,
            y_offset=y_offset,
            pixels=np.zeros((height, width), dtype=np.uint8),
        )


@dataclass
class Light:
    type: int  # LIGHT_HEMI / LIGHT_DIFFUSE / LIGHT_SPECULAR
    shadow: int
    direction: np.ndarray
    intensity: float
