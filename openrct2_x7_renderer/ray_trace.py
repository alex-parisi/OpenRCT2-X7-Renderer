"""
Thin Python wrapper around the native Embree-backed renderer.
"""

__all__ = ["Context", "FinalizedScene", "SceneBuilder", "VIEWS"]

from typing import Any

import numpy as np

from . import _x7_renderer as x7
from .constants import MaterialFlag, TILE_SIZE
from .mesh import Mesh
from .types import IndexedImage, Light

VIEWS: tuple[np.ndarray, ...] = (
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),    # NE
    np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64),   # NW
    np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64),  # SW
    np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),   # SE
)


def _materials_to_dicts(mesh: Mesh) -> list[dict[str, Any]]:
    """Serialize a Mesh's materials into the dict format expected by the native renderer."""
    out: list[dict[str, Any]] = []
    for mat in mesh.materials:
        d: dict[str, Any] = {
            "flags": int(mat.flags),
            "region": int(mat.region),
            "specular_exponent": float(mat.specular_exponent),
            "specular_color": tuple(float(c) for c in mat.specular_color),
            "ambient_color": tuple(float(c) for c in mat.ambient_color),
        }
        if (mat.flags & MaterialFlag.HAS_TEXTURE) and mat.texture is not None:
            d["texture"] = mat.texture.pixels.astype(np.float32, copy=False)
        else:
            d["color"] = tuple(float(c) for c in mat.color)
        out.append(d)
    return out


def _dict_to_image(d: dict[str, Any]) -> IndexedImage:
    """Convert a native renderer image dict to an IndexedImage."""
    return IndexedImage(
        width=int(d["width"]),
        height=int(d["height"]),
        x_offset=int(d["x_offset"]),
        y_offset=int(d["y_offset"]),
        pixels=d["pixels"],
    )


def _add_mesh_to_native(
    inner: x7.Context,
    mesh: Mesh,
    matrix: np.ndarray,
    translation: np.ndarray,
    mask: int,
) -> None:
    """Push a Mesh into the native renderer context."""
    if mesh.faces.shape[0] == 0:
        return
    inner.add_mesh(
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


class FinalizedScene:
    """Scene with committed geometry, ready for rendering.

    Returned by :meth:`SceneBuilder.finalize`. Provides ``render_view``
    and ``render_silhouette`` for producing palette-indexed sprites.

    Can also be used as a context manager; ``end_render`` is called on exit::

        with scene.finalize() as ready:
            img = ready.render_view(view)
    """

    def __init__(self, inner: x7.Context) -> None:
        self._inner = inner

    def __enter__(self) -> "FinalizedScene":
        return self

    def __exit__(self, *_: object) -> None:
        self.end_render()

    def render_view(self, view: np.ndarray) -> IndexedImage:
        """Render the scene under the given view rotation matrix."""
        return _dict_to_image(
            self._inner.render_view(
                view=np.ascontiguousarray(view, dtype=np.float32)
            )
        )

    def render_silhouette(self, view: np.ndarray) -> IndexedImage:
        """Render a solid silhouette of the scene under ``view``.

        Every hit pixel is shaded as flat mid-gray and quantized to the nearest
        RCT2 palette entry; transparent pixels are unchanged.  Useful for
        generating mask sprites.
        """
        return _dict_to_image(
            self._inner.render_silhouette(
                view=np.ascontiguousarray(view, dtype=np.float32)
            )
        )

    def end_render(self) -> None:
        """Release render resources for this pass."""
        self._inner.end_render()


class SceneBuilder:
    """Accumulates models into a scene before rendering.

    Returned by :meth:`Context.begin_render`. Use as a context manager so that
    Embree resources are released even if an exception is raised.

    When :meth:`finalize` has **not** been called (e.g. the block exits early),
    ``__exit__`` calls ``end_render`` automatically.  When :meth:`finalize`
    **has** been called, cleanup ownership passes to the returned
    :class:`FinalizedScene`; use it as a context manager or call its
    :meth:`~FinalizedScene.end_render` explicitly::

        # Pattern A — render and clean up inside the outer with-block
        with ctx.begin_render() as scene:
            with scene.add_model(mesh, mat, t).finalize() as ready:
                img = ready.render_view(view)

        # Pattern B — explicit cleanup
        scene = ctx.begin_render()
        ready = scene.add_model(mesh, rotation, offset).finalize()
        img = ready.render_view(view)
        ready.end_render()
    """

    def __init__(self, inner: x7.Context) -> None:
        self._inner = inner
        self._finalized = False

    def add_model(
        self,
        mesh: Mesh,
        matrix: np.ndarray,
        translation: np.ndarray,
        mask: int = 0,
    ) -> "SceneBuilder":
        """Add a mesh to the scene. Returns ``self`` for chaining."""
        _add_mesh_to_native(self._inner, mesh, matrix, translation, mask)
        return self

    def finalize(self) -> FinalizedScene:
        """Commit the scene geometry and return a renderable scene.

        After this call, ``SceneBuilder.__exit__`` will **not** invoke
        ``end_render``; the returned :class:`FinalizedScene` owns cleanup.
        """
        self._inner.finalize_render()
        self._finalized = True
        return FinalizedScene(self._inner)

    def __enter__(self) -> "SceneBuilder":
        return self

    def __exit__(self, *_: object) -> None:
        if not self._finalized:
            self._inner.end_render()


class Context:
    """Render context wrapping the native Embree-backed renderer.

    Owns the native Embree device and its persistent thread pool.
    Use :meth:`begin_render` to start a render pass::

        ctx = Context(lights=my_lights)
        with ctx.begin_render() as scene:
            ready = scene.add_model(mesh, matrix, translation).finalize()
            img = ready.render_view(VIEWS[0])

    Attributes:
        lights: Light rig used for shading.
        dither: Whether Floyd-Steinberg dithering is applied when quantizing
            to the palette.
        upt: Camera scale as units-per-tile; smaller values zoom in.
    """

    def __init__(
        self, lights: list[Light], dither: bool = True, upt: float = TILE_SIZE
    ) -> None:
        self.lights = lights
        self.dither = dither
        self.upt = upt
        self._inner = x7.Context(lights=lights, dither=dither, upt=upt)

    def __repr__(self) -> str:
        return f"Context(dither={self.dither!r}, upt={self.upt!r}, lights={self.lights!r})"

    def begin_render(self) -> SceneBuilder:
        """Begin a new render pass; returns a :class:`SceneBuilder`."""
        self._inner.begin_render()
        return SceneBuilder(self._inner)
