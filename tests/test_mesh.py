"""Tests for the pure-Python OBJ/MTL loader."""

import numpy as np
from openrct2_x7_renderer.constants import (
    MATERIAL_BACKGROUND_AA,
    MATERIAL_BACKGROUND_AA_DARK,
    MATERIAL_IS_MASK,
    MATERIAL_IS_REMAPPABLE,
    MATERIAL_NO_AO,
    MATERIAL_NO_BLEED,
)
from openrct2_x7_renderer.mesh import (
    Material,
    _classify_material_name,
    load_mesh,
)


def _classify(name):
    mat = Material()
    _classify_material_name(mat, name)
    return mat


def test_remap_region_assignment():
    assert _classify("BodyRemap1").region == 1
    assert _classify("BodyRemap2").region == 2
    assert _classify("BodyRemap3").region == 3
    assert _classify("BodyRemap1").flags & MATERIAL_IS_REMAPPABLE


def test_named_special_regions():
    assert _classify("MyGreyscale").region == 4
    assert _classify("RiderPeep").region == 5


def test_mask_flag():
    plain = _classify("CutoutMask")
    assert plain.flags & MATERIAL_IS_MASK


def test_combined_modifier_flags():
    mat = _classify("ShinyMetal_Edge_NoAO")
    assert mat.flags & MATERIAL_BACKGROUND_AA
    assert mat.flags & MATERIAL_NO_AO


def test_dark_edge_and_no_bleed_flags():
    mat = _classify("Trim_DarkEdge_NoBleed")
    assert mat.flags & MATERIAL_BACKGROUND_AA_DARK
    assert mat.flags & MATERIAL_NO_BLEED


def test_glass_material_sets_is_glass():
    assert _classify("WindowGlass").is_glass
    assert not _classify("Frame").is_glass


def test_front_and_back_wall_side_classification():
    assert _classify("FrontPanel").is_front
    assert not _classify("FrontPanel").is_back
    assert _classify("BackPanel").is_back
    assert not _classify("BackPanel").is_front
    # Untagged faces are shared (neither side) and appear in both wall blocks.
    plain = _classify("Frame")
    assert not plain.is_front
    assert not plain.is_back


def test_back_takes_precedence_over_front_substring():
    # The classifier checks "Back" before "Front" (elif), so a name containing
    # both is treated as a back face -- a behaviour the rear-block split relies
    # on not silently flipping.
    mat = _classify("BackFront")
    assert mat.is_back
    assert not mat.is_front


def _write_obj(tmp_path, body, mtl=None):
    if mtl is not None:
        (tmp_path / "materials.mtl").write_text(mtl)
        body = "mtllib materials.mtl\n" + body
    path = tmp_path / "model.obj"
    path.write_text(body)
    return path


def test_quad_is_fan_triangulated(tmp_path):
    obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n"
    mesh = load_mesh(_write_obj(tmp_path, obj))
    assert mesh.faces.shape == (2, 3)
    assert np.array_equal(mesh.faces[0], [0, 1, 2])
    assert np.array_equal(mesh.faces[1], [0, 2, 3])
    assert mesh.vertices.shape == (4, 3)


def test_generated_normals_when_obj_has_none(tmp_path):
    obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n"
    mesh = load_mesh(_write_obj(tmp_path, obj))
    # Flat quad in the z=0 plane -> all normals point +z.
    assert np.allclose(mesh.normals, [0.0, 0.0, 1.0])


def test_negative_face_indices_resolve_relative(tmp_path):
    # -1 refers to the most recently defined vertex.
    obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n"
    mesh = load_mesh(_write_obj(tmp_path, obj))
    assert mesh.faces.shape == (1, 3)
    assert np.allclose(
        np.sort(mesh.vertices, axis=0), np.sort([[0, 0, 0], [1, 0, 0], [0, 1, 0]], axis=0)
    )


def test_material_order_follows_usemtl(tmp_path):
    mtl = "newmtl Red\nKd 1 0 0\nnewmtl BlueRemap1\nKd 0 0 1\n"
    obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nusemtl BlueRemap1\nf 1 2 3\nusemtl Red\nf 1 2 3\n"
    mesh = load_mesh(_write_obj(tmp_path, obj, mtl))
    # First referenced material is BlueRemap1 -> index 0, remappable region 1.
    assert mesh.materials[0].region == 1
    assert int(mesh.face_materials[0]) == 0
    assert int(mesh.face_materials[1]) == 1


def test_empty_mesh_is_valid_and_degenerate(tmp_path):
    obj = "v 0 0 0\nv 1 0 0\n"  # no faces
    mesh = load_mesh(_write_obj(tmp_path, obj))
    assert mesh.faces.shape == (0, 3)
    assert mesh.vertices.shape == (0, 3)
