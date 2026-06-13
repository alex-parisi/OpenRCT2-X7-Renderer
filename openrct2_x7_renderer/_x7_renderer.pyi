"""Type stubs for the _x7_renderer pybind11/Embree C extension."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

def palette_rgb() -> NDArray[np.uint8]:
    """Return the native RCT2 palette as a (256, 3) uint8 array."""
    ...

MATERIAL_HAS_TEXTURE: int
MATERIAL_IS_REMAPPABLE: int
MATERIAL_IS_MASK: int
MATERIAL_NO_AO: int
MATERIAL_BACKGROUND_AA: int
MATERIAL_BACKGROUND_AA_DARK: int
MATERIAL_IS_VISIBLE_MASK: int
MATERIAL_NO_BLEED: int
MATERIAL_IS_FLAT_SHADED: int


MESH_MASK: int
MESH_GHOST: int


LIGHT_HEMI: int
LIGHT_DIFFUSE: int
LIGHT_SPECULAR: int


DITHER_NONE: int
DITHER_FLOYD_STEINBERG: int
DITHER_BAYER: int
DITHER_BLUE_NOISE: int


class Context:
    """Native Embree render context.

    Typical lifecycle::

        ctx = Context(lights=lights, dither_mode=DITHER_FLOYD_STEINBERG, stability=0.0, upt=32.0)
        ctx.begin_render()
        ctx.add_mesh(...)           # repeat for each mesh
        ctx.finalize_render()
        img = ctx.render_view(view_matrix)
        ctx.end_render()

    The Python wrapper (``ray_trace.Context``) exposes a typestate API
    on top of this: ``begin_render`` → ``SceneBuilder`` → ``FinalizedScene``.
    """

    def __init__(
        self,
        lights: list[Any],
        dither_mode: int,
        stability: float,
        upt: float,
    ) -> None:
        """
        Args:
            lights: List of light objects.  Each must expose the attributes
                    ``type`` (int), ``shadow`` (bool), ``intensity`` (float),
                    and ``direction`` (float32 array of shape ``(3,)``).
            dither_mode: Quantisation dithering, one of ``DITHER_NONE``,
                    ``DITHER_FLOYD_STEINBERG`` (highest fidelity, temporally
                    unstable across animation frames), ``DITHER_BAYER``
                    (screen-anchored ordered dither, stable across frames), or
                    ``DITHER_BLUE_NOISE`` (screen-anchored blue-noise tile;
                    like Bayer but with less perceptible residual motion).
            stability: Temporal-stability deadband in 8-bit sRGB units. Snaps the
                    pre-dither colour onto a grid of this size so sub-step shading
                    changes quantise identically between frames. ``0`` disables it.
            upt:    Units-per-tile scale factor used by the renderer kernel.
        """
        ...

    def begin_render(self) -> None:
        """Open a new scene.  Must be called before :meth:`add_mesh`."""
        ...

    def add_mesh(
        self,
        vertices: NDArray[np.float32],
        normals: NDArray[np.float32],
        uvs: NDArray[np.float32],
        faces: NDArray[np.uint32],
        face_materials: NDArray[np.uint32],
        materials: list[dict[str, Any]],
        matrix: NDArray[np.float32],
        translation: NDArray[np.float32],
        mask: int = 0,
    ) -> None:
        """Upload one mesh to the Embree scene.

        Must be called between :meth:`begin_render` and
        :meth:`finalize_render`.

        Args:
            vertices:      ``(N, 3)`` float32 vertex positions in model space.
            normals:       ``(N, 3)`` float32 per-vertex normals (unit length).
            uvs:           ``(N, 2)`` float32 UV texture coordinates.
            faces:         ``(F, 3)`` uint32 triangle index triples into *vertices*.
            face_materials: ``(F,)`` uint32 per-face material indices.
            materials:     List of material dicts, one per unique material.
                           Required keys: ``flags`` (int), ``region`` (int),
                           ``specular_exponent`` (float),
                           ``specular_color`` (3-sequence of float),
                           ``ambient_color``   (3-sequence of float).
                           When ``flags & MATERIAL_HAS_TEXTURE``: include
                           ``texture`` (float32 array of shape ``(H, W, 3)``).
                           Otherwise: include ``color`` (3-sequence of float).
            matrix:        ``(3, 3)`` float32 rotation / basis matrix.
            translation:   ``(3,)``  float32 translation vector.
            mask:          Combination of ``MESH_MASK`` / ``MESH_GHOST``
                           flag bits (default ``0`` = normal solid mesh).
        """
        ...

    def finalize_render(self) -> None:
        """Commit all uploaded meshes into the Embree BVH.

        Call once after all :meth:`add_mesh` invocations and before any
        :meth:`render_view` / :meth:`render_silhouette` calls.
        """
        ...

    def render_view(self, view: NDArray[np.float32]) -> dict[str, Any]:
        """Ray-trace the scene and return a palette-indexed image.

        Releases the GIL during the ray-tracing hot path.

        Args:
            view: ``(3, 3)`` float32 orthographic view / rotation matrix.

        Returns:
            Dict with the following keys:

            * ``"width"``    – ``int`` image width in pixels.
            * ``"height"``   – ``int`` image height in pixels.
            * ``"x_offset"`` – ``int`` left offset relative to the tile origin.
            * ``"y_offset"`` – ``int`` top  offset relative to the tile origin.
            * ``"pixels"``   – ``NDArray[np.uint8]`` of shape ``(height, width)``;
              palette indices where ``0`` is transparent.
        """
        ...

    def render_silhouette(self, view: NDArray[np.float32]) -> dict[str, Any]:
        """Like :meth:`render_view` but renders geometry as flat mid-gray (0.5, 0.5, 0.5)
        and quantizes to the nearest palette entry, producing a solid silhouette sprite.

        Useful for generating mask sprites.

        Returns:
            Same dict schema as :meth:`render_view`.
        """
        ...

    def end_render(self) -> None:
        """Destroy the Embree scene and release all mesh data.

        Safe to call even if :meth:`begin_render` was never called.
        After this the instance may be reused by calling
        :meth:`begin_render` again.
        """
        ...
