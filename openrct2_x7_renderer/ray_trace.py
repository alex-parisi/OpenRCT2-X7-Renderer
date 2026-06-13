"""
Thin Python wrapper around the native Embree-backed renderer.
"""

__all__ = ["Context", "FinalizedScene", "SceneBuilder", "VIEWS"]

from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import _x7_renderer as x7
from .constants import TILE_SIZE, MaterialFlag
from .mesh import Mesh
from .remap import apply_remap_overrides
from .types import IndexedImage, Light

VIEWS: tuple[NDArray[np.float64], ...] = (
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),    # NE
    np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64),   # NW
    np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64),  # SW
    np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),   # SE
)

# Palette dithering modes accepted by Context. ``floyd_steinberg`` gives the
# highest fidelity but is temporally unstable (its error diffusion makes the
# pattern "swim" across an animation's frames); ``bayer`` is a screen-anchored
# ordered dither whose output is identical between frames wherever the colour
# is unchanged, ideal for animated objects. ``blue_noise`` is the same
# screen-anchored ordered dither but driven by a 64x64 void-and-cluster tile
# instead of the 8x8 Bayer matrix, so the residual pattern motion wherever the
# shading does change (rotation, animation) is far less perceptible.
DITHER_MODES: dict[str, int] = {
    "none": x7.DITHER_NONE,
    "floyd_steinberg": x7.DITHER_FLOYD_STEINBERG,
    "bayer": x7.DITHER_BAYER,
    "blue_noise": x7.DITHER_BLUE_NOISE,
}


def _resolve_dither(dither: bool | str) -> str:
    """Normalize a ``dither`` argument to a canonical mode name.

    Accepts a mode string (``"none"`` / ``"floyd_steinberg"`` / ``"bayer"`` /
    ``"blue_noise"``) or, for backwards compatibility, a bool where ``True``
    selects Floyd-Steinberg and ``False`` disables dithering.
    """
    if isinstance(dither, bool):
        return "floyd_steinberg" if dither else "none"
    if dither not in DITHER_MODES:
        raise ValueError(
            f"unknown dither mode {dither!r}; expected one of {sorted(DITHER_MODES)}"
        )
    return dither


def _materials_to_dicts(mesh: Mesh) -> list[dict[str, Any]]:
    """Serialize a Mesh's materials into the dict format expected by the native renderer."""
    out: list[dict[str, Any]] = []
    for mat in mesh.materials:
        # The native renderer reads "texture" whenever HAS_TEXTURE is set, so a
        # material carrying the flag without an actual texture must have the
        # flag cleared rather than crash the C++ side with a missing key.
        texture = mat.texture if mat.flags & MaterialFlag.HAS_TEXTURE else None
        flags = int(mat.flags)
        if texture is None:
            flags &= ~MaterialFlag.HAS_TEXTURE
        d: dict[str, Any] = {
            "flags": flags,
            "region": int(mat.region),
            "specular_exponent": float(mat.specular_exponent),
            "specular_color": tuple(float(c) for c in mat.specular_color),
            "ambient_color": tuple(float(c) for c in mat.ambient_color),
        }
        if texture is not None:
            d["texture"] = texture.pixels.astype(np.float32, copy=False)
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
    matrix: NDArray[np.float64],
    translation: NDArray[np.float64],
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

    def __init__(
        self, inner: x7.Context, remap_overrides: dict[int, tuple[int, ...]] | None = None
    ) -> None:
        self._inner = inner
        self._remap_overrides = remap_overrides or {}
        self._ended = False

    def __enter__(self) -> "FinalizedScene":
        return self

    def __exit__(self, *_: object) -> None:
        self.end_render()

    def render_view(self, view: NDArray[np.float64]) -> IndexedImage:
        """Render the scene under the given view rotation matrix.

        When the owning :class:`Context` was given ``remap_overrides`` (the
        preview-only ``--test`` recolouring), the remap windows in the result
        are rewritten to the chosen colour ramps; otherwise the image is
        returned exactly as the renderer produced it.
        """
        image = _dict_to_image(
            self._inner.render_view(
                view=np.ascontiguousarray(view, dtype=np.float32)
            )
        )
        if not self._remap_overrides:
            return image
        return replace(
            image, pixels=apply_remap_overrides(image.pixels, self._remap_overrides)
        )

    def render_silhouette(self, view: NDArray[np.float64]) -> IndexedImage:
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
        """Release render resources for this pass (idempotent)."""
        if self._ended:
            return
        self._ended = True
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

    def __init__(
        self, inner: x7.Context, remap_overrides: dict[int, tuple[int, ...]] | None = None
    ) -> None:
        self._inner = inner
        self._remap_overrides = remap_overrides or {}
        self._finalized = False

    def add_model(
        self,
        mesh: Mesh,
        matrix: NDArray[np.float64],
        translation: NDArray[np.float64],
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
        return FinalizedScene(self._inner, self._remap_overrides)

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
        dither_mode: Palette dithering mode (``"none"`` / ``"floyd_steinberg"``
            / ``"bayer"`` / ``"blue_noise"``). Use ``"bayer"`` or ``"blue_noise"``
            for animated objects so the dither pattern stays stable across frames;
            ``"blue_noise"`` further reduces the residual motion under rotation.
        stability: Temporal-stability deadband in 8-bit sRGB units (0 disables).
            Snaps the pre-dither colour onto a coarser grid so shading changes
            smaller than the deadband quantise identically across frames,
            suppressing residual "swimming"; the dither masks the banding.
        upt: Camera scale as units-per-tile; smaller values zoom in.
        remap_overrides: Preview-only ``{region: ramp}`` recolouring applied to
            ``render_view`` output (see :mod:`~openrct2_x7_renderer.remap`).
            Empty by default, so normal renders keep their raw remap windows.
    """

    def __init__(
        self,
        lights: list[Light],
        dither: bool | str = True,
        upt: float = TILE_SIZE,
        remap_overrides: dict[int, tuple[int, ...]] | None = None,
        stability: float = 0.0,
    ) -> None:
        self.lights = lights
        self.dither_mode = _resolve_dither(dither)
        self.stability = float(stability)
        self.upt = upt
        self.remap_overrides = remap_overrides or {}
        self._inner = x7.Context(
            lights=lights,
            dither_mode=DITHER_MODES[self.dither_mode],
            stability=self.stability,
            upt=upt,
        )

    def __repr__(self) -> str:
        return (
            f"Context(dither_mode={self.dither_mode!r}, stability={self.stability!r}, "
            f"upt={self.upt!r}, lights={self.lights!r})"
        )

    def begin_render(self) -> SceneBuilder:
        """Begin a new render pass; returns a :class:`SceneBuilder`."""
        self._inner.begin_render()
        return SceneBuilder(self._inner, self.remap_overrides)
