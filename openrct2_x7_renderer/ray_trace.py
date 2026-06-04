"""
Thin Python wrapper around the native Embree-backed renderer.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from . import _x7_renderer  # type: ignore[attr-defined]  # C extension, no stubs
from .constants import MATERIAL_HAS_TEXTURE, TILE_SIZE
from .mesh import Mesh
from .types import IndexedImage, Light


def rotate_x(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rotate_y(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rotate_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


# Four-corner views (mirrors Renderer.cpp `views`).
VIEWS = [
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
    np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64),
    np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64),
    np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),
]


def _materials_to_dicts(mesh: Mesh) -> list[dict]:
    out: list[dict] = []
    for mat in mesh.materials:
        d: dict = {
            "flags": int(mat.flags),
            "region": int(mat.region),
            "specular_exponent": float(mat.specular_exponent),
            "specular_color": tuple(float(c) for c in mat.specular_color),
            "ambient_color": tuple(float(c) for c in mat.ambient_color),
        }
        if (mat.flags & MATERIAL_HAS_TEXTURE) and mat.texture is not None:
            d["texture"] = mat.texture.pixels.astype(np.float32, copy=False)
        else:
            d["color"] = tuple(float(c) for c in mat.color)
        out.append(d)
    return out


@dataclass
class Context:
    """Lifetime manager for the native renderer."""

    lights: list[Light]
    dither: bool
    upt: float
    _inner: _x7_renderer.Context = field(repr=False)

    @classmethod
    def make(cls, lights: list[Light], dither: bool = True, upt: float = TILE_SIZE) -> "Context":
        inner = _x7_renderer.Context(lights=lights, dither=dither, upt=upt)
        return cls(lights=lights, dither=dither, upt=upt, _inner=inner)

    def begin_render(self) -> None:
        self._inner.begin_render()

    def add_model(
        self, mesh: Mesh, matrix: np.ndarray, translation: np.ndarray, mask: int = 0
    ) -> None:
        if mesh.faces.shape[0] == 0:
            return
        self._inner.add_mesh(
            vertices=mesh.vertices.astype(np.float32, copy=False),
            normals=mesh.normals.astype(np.float32, copy=False),
            uvs=mesh.uvs.astype(np.float32, copy=False),
            faces=mesh.faces.astype(np.uint32, copy=False),
            face_materials=mesh.face_materials.astype(np.uint32, copy=False),
            materials=_materials_to_dicts(mesh),
            matrix=np.ascontiguousarray(matrix, dtype=np.float32),
            translation=np.ascontiguousarray(translation, dtype=np.float32),
            mask=int(mask),
        )

    def finalize_render(self) -> None:
        self._inner.finalize_render()

    def end_render(self) -> None:
        self._inner.end_render()


def _dict_to_image(d: dict) -> IndexedImage:
    return IndexedImage(
        width=int(d["width"]),
        height=int(d["height"]),
        x_offset=int(d["x_offset"]),
        y_offset=int(d["y_offset"]),
        pixels=d["pixels"],
    )


def render_view(context: Context, view: np.ndarray) -> IndexedImage:
    """Render the current scene under `view`.

    The native renderer manages its own random state (AO is seeded
    per-pixel from a hit-position hash), so there is no seed to thread
    through from Python.
    """
    return _dict_to_image(
        context._inner.render_view(view=np.ascontiguousarray(view, dtype=np.float32))
    )


def render_silhouette(context: Context, view: np.ndarray) -> IndexedImage:
    return _dict_to_image(
        context._inner.render_silhouette(view=np.ascontiguousarray(view, dtype=np.float32))
    )
