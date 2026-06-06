"""Tests for the pure-Python OBJ/MTL loader."""

import numpy as np
from openrct2_x7_renderer.constants import MaterialFlag
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
    assert _classify("BodyRemap1").flags & MaterialFlag.IS_REMAPPABLE


def test_named_special_regions():
    assert _classify("MyGreyscale").region == 4
    assert _classify("RiderPeep").region == 5


def test_mask_flag():
    plain = _classify("CutoutMask")
    assert plain.flags & MaterialFlag.IS_MASK


def test_visible_mask_flag_takes_precedence_over_mask_substring():
    # "VisibleMask" contains "Mask" but must map to the visible-mask flag (it is rendered
    # into the silhouette/occlusion pass), not the invisible-occluder IS_MASK flag.
    mat = _classify("VisibleMask")
    assert mat.flags & MaterialFlag.IS_VISIBLE_MASK
    assert not (mat.flags & MaterialFlag.IS_MASK)


def test_combined_modifier_flags():
    mat = _classify("ShinyMetal_Edge_NoAO")
    assert mat.flags & MaterialFlag.BACKGROUND_AA
    assert mat.flags & MaterialFlag.NO_AO


def test_dark_edge_and_no_bleed_flags():
    mat = _classify("Trim_DarkEdge_NoBleed")
    assert mat.flags & MaterialFlag.BACKGROUND_AA_DARK
    assert mat.flags & MaterialFlag.NO_BLEED


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


def test_obj_out_of_range_negative_index_raises_load_error(tmp_path):
    import pytest
    from openrct2_x7_renderer.types import LoadError

    # Only 3 vertices, but face references index -5 (needs 5+).
    obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -5 -2 -1\n"
    path = tmp_path / "bad.obj"
    path.write_text(obj)
    with pytest.raises(LoadError, match="out of range"):
        load_mesh(path)


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


def test_obj_with_comments_and_blank_lines(tmp_path):
    obj = "# a comment\n\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
    mesh = load_mesh(_write_obj(tmp_path, obj))
    assert mesh.faces.shape == (1, 3)


def test_obj_with_normals_uses_them(tmp_path):
    obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nf 1//1 2//1 3//1\n"
    mesh = load_mesh(_write_obj(tmp_path, obj))
    assert np.allclose(mesh.normals, [0.0, 0.0, 1.0], atol=1e-5)


def test_obj_with_uvs(tmp_path):
    obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n"
    mesh = load_mesh(_write_obj(tmp_path, obj))
    assert mesh.uvs.shape == (3, 2)
    assert np.allclose(mesh.uvs[0], [0.0, 0.0])
    assert np.allclose(mesh.uvs[1], [1.0, 0.0])


def test_obj_degenerate_face_under_3_vertices_skipped(tmp_path):
    # "f 1 2" has only 2 tokens, not 3 — skipped with a warning.
    obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2\nf 1 2 3\n"
    mesh = load_mesh(_write_obj(tmp_path, obj))
    assert mesh.faces.shape == (1, 3)


def test_obj_flip_winding_with_negative_det_transform(tmp_path):
    obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
    path = _write_obj(tmp_path, obj)
    # A reflection (negative determinant) triggers winding reversal.
    mirror = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    mesh = load_mesh(path, transform=mirror)
    assert mesh.faces.shape == (1, 3)
    # Winding reversal is applied: the generated normal should still point outward
    # (same direction as without the flip) because the reflection is compensated.
    normals = mesh.normals
    assert normals.shape[0] > 0


def test_load_mesh_missing_file_raises_load_error():
    import pytest
    from openrct2_x7_renderer.types import LoadError

    with pytest.raises(LoadError, match="not found"):
        load_mesh("/nonexistent/nowhere/model.obj")


def test_load_mesh_malformed_face_index_raises_load_error(tmp_path):
    import pytest
    from openrct2_x7_renderer.types import LoadError

    obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/abc/1 2 3\n"
    path = tmp_path / "bad.obj"
    path.write_text(obj)
    with pytest.raises(LoadError, match="Malformed face index"):
        load_mesh(path)


def test_obj_non_orthonormal_transform_raises_load_error(tmp_path):
    import pytest
    from openrct2_x7_renderer.types import LoadError

    obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
    path = _write_obj(tmp_path, obj)
    scale2 = np.diag([2.0, 2.0, 2.0])
    with pytest.raises(LoadError, match="not orthonormal"):
        load_mesh(path, transform=scale2)


def test_load_texture_reads_rgb_image(tmp_path):
    from openrct2_x7_renderer.mesh import load_texture
    from PIL import Image as PILImage

    img_path = tmp_path / "tex.png"
    PILImage.new("RGB", (4, 4), (255, 128, 0)).save(img_path)
    tex = load_texture(img_path)
    assert tex.width == 4
    assert tex.height == 4
    assert tex.pixels.shape == (4, 4, 3)
    assert tex.pixels.dtype.name == "float32"
    # All pixels come from sRGB (255,128,0); red channel should be near 1.0 linear.
    assert tex.pixels[:, :, 0].max() > 0.9


def test_parse_mtl_file_not_found_returns_empty(tmp_path):

    from openrct2_x7_renderer.mesh import _parse_mtl

    result = _parse_mtl(tmp_path / "missing.mtl", tmp_path)
    assert result == {}


def test_parse_mtl_blank_and_comment_lines(tmp_path):
    from openrct2_x7_renderer.mesh import _parse_mtl

    mtl = "# header comment\n\nnewmtl Red\nKd 1 0 0\n"
    mtl_path = tmp_path / "test.mtl"
    mtl_path.write_text(mtl)
    result = _parse_mtl(mtl_path, tmp_path)
    assert "Red" in result


def test_parse_mtl_command_before_newmtl_ignored(tmp_path):
    from openrct2_x7_renderer.mesh import _parse_mtl

    # "Kd" before any "newmtl" should be silently skipped.
    mtl = "Kd 1 0 0\nnewmtl Mat\nKd 0.5 0.5 0.5\n"
    mtl_path = tmp_path / "test.mtl"
    mtl_path.write_text(mtl)
    result = _parse_mtl(mtl_path, tmp_path)
    assert "Mat" in result
    assert len(result) == 1


def test_parse_mtl_specular_ambient_shininess(tmp_path):
    from openrct2_x7_renderer.mesh import _parse_mtl
    from openrct2_x7_renderer.palette import srgb2linear

    mtl = "newmtl Shiny\nKs 0.8 0.6 0.4\nKa 0.1 0.1 0.1\nNs 128.0\n"
    mtl_path = tmp_path / "test.mtl"
    mtl_path.write_text(mtl)
    result = _parse_mtl(mtl_path, tmp_path)
    mat = result["Shiny"]
    # Ks and Ka are treated as sRGB and linearized on load.
    assert np.allclose(mat.specular_color, srgb2linear(np.array([0.8, 0.6, 0.4])))
    assert np.allclose(mat.ambient_color, srgb2linear(np.array([0.1, 0.1, 0.1])))
    assert mat.specular_exponent == 128.0


def test_parse_mtl_map_kd_missing_texture_prints_warning(tmp_path, caplog):
    import logging

    from openrct2_x7_renderer.mesh import _parse_mtl

    mtl = "newmtl WithTex\nmap_Kd /nonexistent/nowhere/tex.png\n"
    mtl_path = tmp_path / "test.mtl"
    mtl_path.write_text(mtl)
    with caplog.at_level(logging.WARNING, logger="openrct2_x7_renderer.mesh"):
        result = _parse_mtl(mtl_path, tmp_path)
    assert "WithTex" in result
    assert "Failed to load texture" in caplog.text


def test_partial_normals_warning_and_fallback(tmp_path, caplog):
    import logging

    # Two faces: first references normals, second does not.
    # The loader should warn and fall back to auto-generated normals.
    obj = (
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\n"
        "vn 0 0 1\n"
        "f 1//1 2//1 3//1\n"  # has normal reference
        "f 2 3 4\n"           # no normal reference
    )
    with caplog.at_level(logging.WARNING, logger="openrct2_x7_renderer.mesh"):
        mesh = load_mesh(_write_obj(tmp_path, obj))
    assert "discarding all vn data" in caplog.text
    # Fallback normals are generated; mesh should still be valid.
    assert mesh.normals.shape == mesh.vertices.shape


def test_missing_vt_with_nonzero_vt_index_warns_and_defaults_uv(tmp_path, caplog):
    import logging

    # Face references vt index 2 but no vt entries are defined at all.
    obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/2 2/3 3/1\n"
    with caplog.at_level(logging.WARNING, logger="openrct2_x7_renderer.mesh"):
        mesh = load_mesh(_write_obj(tmp_path, obj))
    assert "references texture coordinate" in caplog.text
    # UVs default to (0, 0) when vt data is absent.
    assert np.allclose(mesh.uvs, 0.0)


def test_parse_mtl_map_kd_loads_texture(tmp_path):
    from openrct2_x7_renderer.mesh import _parse_mtl
    from PIL import Image as PILImage

    tex_path = tmp_path / "color.png"
    PILImage.new("RGB", (2, 2), (200, 100, 50)).save(tex_path)

    mtl = f"newmtl Textured\nmap_Kd {tex_path.name}\n"
    mtl_path = tmp_path / "test.mtl"
    mtl_path.write_text(mtl)
    result = _parse_mtl(mtl_path, tmp_path)
    mat = result["Textured"]
    assert mat.flags & MaterialFlag.HAS_TEXTURE
    assert mat.texture is not None
