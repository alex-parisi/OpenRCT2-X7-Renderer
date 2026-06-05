"""Tests for the thin Python wrapper around the native Embree renderer."""

import math

import numpy as np
from openrct2_x7_renderer.constants import MaterialFlag
from openrct2_x7_renderer.geometry import rotate_x, rotate_y, rotate_z
from openrct2_x7_renderer.lights import default_lights
from openrct2_x7_renderer.mesh import Material, Mesh, Texture
from openrct2_x7_renderer.ray_trace import (
    VIEWS,
    Context,
    _materials_to_dicts,
)


def _empty_context():
    return Context(lights=default_lights(), dither=False, upt=32.0)


def _simple_mesh():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    n = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
    uvs = np.zeros((3, 2), dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    fm = np.array([0], dtype=np.uint32)
    return Mesh(
        vertices=v, normals=n, uvs=uvs, faces=faces, face_materials=fm, materials=[Material()]
    )


# ---------- rotation matrices ----------


def test_rotate_x_identity_at_zero():
    assert np.allclose(rotate_x(0.0), np.eye(3))


def test_rotate_x_90_degrees():
    rot = rotate_x(math.pi / 2)
    # +Y maps to +Z.
    result = rot @ np.array([0, 1, 0])
    assert np.allclose(result, [0, 0, 1], atol=1e-6)


def test_rotate_y_identity_at_zero():
    assert np.allclose(rotate_y(0.0), np.eye(3))


def test_rotate_y_90_degrees():
    rot = rotate_y(math.pi / 2)
    # +X maps to -Z (right-hand rule about +Y).
    result = rot @ np.array([1, 0, 0])
    assert np.allclose(result, [0, 0, -1], atol=1e-6)


def test_rotate_z_identity_at_zero():
    assert np.allclose(rotate_z(0.0), np.eye(3))


def test_rotate_z_90_degrees():
    rot = rotate_z(math.pi / 2)
    # +X maps to +Y.
    result = rot @ np.array([1, 0, 0])
    assert np.allclose(result, [0, 1, 0], atol=1e-6)


def test_views_are_four_rotation_matrices():
    assert len(VIEWS) == 4
    for v in VIEWS:
        assert v.shape == (3, 3)
        # Each view matrix must be orthonormal (rotation).
        assert np.allclose(v @ v.T, np.eye(3), atol=1e-6)


# ---------- _materials_to_dicts ----------


def test_materials_to_dicts_plain_color():
    mat = Material()
    mat.color = np.array([0.8, 0.4, 0.2])
    mesh = Mesh(
        vertices=np.zeros((3, 3), dtype=np.float32),
        normals=np.zeros((3, 3), dtype=np.float32),
        uvs=np.zeros((3, 2), dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.uint32),
        face_materials=np.array([0], dtype=np.uint32),
        materials=[mat],
    )
    result = _materials_to_dicts(mesh)
    assert len(result) == 1
    assert "color" in result[0]
    assert "texture" not in result[0]
    assert np.allclose(result[0]["color"], (0.8, 0.4, 0.2))


def test_materials_to_dicts_with_texture():
    mat = Material()
    mat.flags = MaterialFlag.HAS_TEXTURE
    mat.texture = Texture(width=2, height=2, pixels=np.ones((2, 2, 3), dtype=np.float32))
    mesh = Mesh(
        vertices=np.zeros((3, 3), dtype=np.float32),
        normals=np.zeros((3, 3), dtype=np.float32),
        uvs=np.zeros((3, 2), dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.uint32),
        face_materials=np.array([0], dtype=np.uint32),
        materials=[mat],
    )
    result = _materials_to_dicts(mesh)
    assert "texture" in result[0]
    assert "color" not in result[0]


def test_materials_to_dicts_empty_materials():
    mesh = Mesh.empty()
    assert _materials_to_dicts(mesh) == []


# ---------- Context lifecycle ----------


def test_context_direct_init():
    lights = default_lights()
    ctx = Context(lights=lights, dither=True, upt=3.3)
    assert isinstance(ctx, Context)
    assert ctx.dither is True
    assert ctx.upt == 3.3
    assert ctx.lights is lights


def test_context_constructor_returns_context():
    ctx = _empty_context()
    assert isinstance(ctx, Context)
    assert ctx.dither is False
    assert ctx.upt == 32.0


def test_context_repr_contains_key_attributes():
    ctx = Context(lights=default_lights(), dither=True, upt=5.0)
    r = repr(ctx)
    assert "Context(" in r
    assert "dither=True" in r
    assert "upt=5.0" in r


def test_context_full_lifecycle_empty_scene():
    ctx = _empty_context()
    with ctx.begin_render() as scene:
        ready = scene.finalize()
        img = ready.render_view(VIEWS[0])
    assert img.width >= 1
    assert img.height >= 1


def test_context_add_model_empty_mesh_is_noop():
    ctx = _empty_context()
    with ctx.begin_render() as scene:
        scene.add_model(Mesh.empty(), np.eye(3, dtype=np.float64), np.zeros(3))
        scene.finalize()


def test_context_add_model_with_geometry():
    ctx = _empty_context()
    with ctx.begin_render() as scene:
        scene.add_model(_simple_mesh(), np.eye(3, dtype=np.float64), np.zeros(3))
        scene.finalize()


def test_context_add_model_with_mask():
    ctx = _empty_context()
    with ctx.begin_render() as scene:
        scene.add_model(_simple_mesh(), np.eye(3, dtype=np.float64), np.zeros(3), mask=1)
        scene.finalize()


# ---------- render_view and render_silhouette ----------


def test_render_view_returns_indexed_image():
    ctx = _empty_context()
    with ctx.begin_render() as scene:
        ready = scene.finalize()
        img = ready.render_view(VIEWS[0])
    from openrct2_x7_renderer.types import IndexedImage

    assert isinstance(img, IndexedImage)
    assert img.pixels.dtype == np.uint8


def _remappable_mesh(region: int):
    mat = Material()
    mat.flags = MaterialFlag.IS_REMAPPABLE
    mat.region = region
    mesh = _simple_mesh()
    mesh.materials = [mat]
    return mesh


def test_render_view_recolours_remap_window_when_overrides_set():
    from openrct2_x7_renderer.remap import REMAP_COLOR_RAMPS, REMAP_WINDOWS

    ramp = REMAP_COLOR_RAMPS["bordeaux_red"]
    ctx = Context(
        lights=default_lights(), dither=False, upt=32.0, remap_overrides={1: ramp}
    )
    with ctx.begin_render() as scene:
        scene.add_model(_remappable_mesh(1), np.eye(3, dtype=np.float64), np.zeros(3))
        img = scene.finalize().render_view(VIEWS[0])

    lit = img.pixels[img.pixels != 0]
    assert lit.size > 0  # the triangle rendered something
    window = set(range(REMAP_WINDOWS[1], REMAP_WINDOWS[1] + 12))
    # No raw remap-window indices survive; every lit pixel is from the target ramp.
    assert window.isdisjoint(set(lit.tolist()))
    assert set(lit.tolist()).issubset(set(ramp))


def test_render_view_leaves_remap_window_when_no_overrides():
    from openrct2_x7_renderer.remap import REMAP_WINDOWS

    ctx = _empty_context()  # no remap_overrides
    with ctx.begin_render() as scene:
        scene.add_model(_remappable_mesh(1), np.eye(3, dtype=np.float64), np.zeros(3))
        img = scene.finalize().render_view(VIEWS[0])

    lit = img.pixels[img.pixels != 0]
    window = set(range(REMAP_WINDOWS[1], REMAP_WINDOWS[1] + 12))
    # Untouched: a remappable material renders into its raw remap window.
    assert set(lit.tolist()).issubset(window)


def test_render_silhouette_returns_indexed_image():
    ctx = _empty_context()
    with ctx.begin_render() as scene:
        ready = scene.finalize()
        img = ready.render_silhouette(VIEWS[0])
    from openrct2_x7_renderer.types import IndexedImage

    assert isinstance(img, IndexedImage)


def test_render_all_four_views():
    ctx = _empty_context()
    with ctx.begin_render() as scene:
        ready = scene.finalize()
        for view in VIEWS:
            img = ready.render_view(view)
            assert img.width >= 1


def test_context_as_context_manager():
    ctx = Context(lights=default_lights(), dither=False, upt=32.0)
    with ctx.begin_render() as scene:
        ready = scene.finalize()
        img = ready.render_view(VIEWS[0])
        assert img.width >= 1


def test_finalized_scene_explicit_end_render():
    ctx = _empty_context()
    scene = ctx.begin_render()
    ready = scene.finalize()
    ready.render_view(VIEWS[0])
    ready.end_render()


def test_finalized_scene_as_context_manager():
    # FinalizedScene.__enter__ / __exit__ — end_render is called automatically.
    ctx = _empty_context()
    with ctx.begin_render() as builder:
        with builder.finalize() as ready:
            img = ready.render_view(VIEWS[0])
    assert img.width >= 1


def test_scene_builder_exit_without_finalize_calls_end_render():
    # SceneBuilder.__exit__ when finalize() was never called must still
    # clean up the scene so begin_render() can be called again.
    ctx = _empty_context()
    with ctx.begin_render():
        pass  # __exit__ hits the `if not self._finalized` branch
    # Renderer must be usable again after the cleanup.
    with ctx.begin_render() as scene:
        scene.add_model(_simple_mesh(), np.eye(3, dtype=np.float64), np.zeros(3))
        scene.finalize().end_render()
