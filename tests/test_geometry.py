"""Tests for the multi-tile (large scenery) geometry baking utilities."""

import numpy as np
from openrct2_x7_renderer.constants import MeshFlag
from openrct2_x7_renderer.geometry import (
    assign_faces_to_tiles,
    clip_mesh_to_tile,
    combine_model_world,
    face_centroids,
    split_mesh_by_ghost,
    subset_mesh,
)
from openrct2_x7_renderer.mesh import Material, Mesh
from openrct2_x7_renderer.types import MeshFrame, Model


def _tri_mesh(verts, material_name="Mat"):
    """A single-triangle mesh at the given three vertices."""
    v = np.array(verts, dtype=np.float32)
    return Mesh(
        vertices=v,
        normals=np.tile([0.0, 0.0, 1.0], (3, 1)).astype(np.float32),
        uvs=np.zeros((3, 2), dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.uint32),
        face_materials=np.array([0], dtype=np.uint32),
        materials=[Material()],
    )


def _model(*placements):
    return Model(meshes=[list(p) for p in placements])


def test_combine_translates_placement():
    mesh = _tri_mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    frame = MeshFrame(mesh_index=0, position=np.array([10.0, 0.0, 0.0]))
    out = combine_model_world([mesh], _model([frame]))
    assert np.allclose(out.vertices, [[10, 0, 0], [11, 0, 0], [10, 1, 0]])


def test_combine_rotates_about_y():
    # orientation[0] -> rotate_y; 90° maps +X (forward) to -Z.
    mesh = _tri_mesh([[1, 0, 0], [2, 0, 0], [1, 1, 0]])
    frame = MeshFrame(mesh_index=0, orientation=np.array([90.0, 0.0, 0.0]))
    out = combine_model_world([mesh], _model([frame]))
    # rot_y(90): (x,y,z) -> (z, y, -x); so [1,0,0] -> [0,0,-1].
    assert np.allclose(out.vertices[0], [0, 0, -1], atol=1e-6)
    assert np.allclose(out.vertices[1], [0, 0, -2], atol=1e-6)


def test_combine_concatenates_and_offsets_indices():
    a = _tri_mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    b = _tri_mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    out = combine_model_world([a, b], _model([MeshFrame(0)], [MeshFrame(1)]))
    assert out.vertices.shape == (6, 3)
    assert out.faces.shape == (2, 3)
    # Second face must reference the second mesh's vertex block.
    assert np.array_equal(out.faces[1], [3, 4, 5])
    # Material offset applied to the second placement's face_materials.
    assert int(out.face_materials[1]) == 1
    assert len(out.materials) == 2


def test_combine_skips_empty_and_unset_meshes():
    mesh = _tri_mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    empty = Mesh(
        vertices=np.zeros((0, 3), np.float32),
        normals=np.zeros((0, 3), np.float32),
        uvs=np.zeros((0, 2), np.float32),
        faces=np.zeros((0, 3), np.uint32),
        face_materials=np.zeros((0,), np.uint32),
        materials=[],
    )
    out = combine_model_world(
        [mesh, empty],
        _model([MeshFrame(mesh_index=-1)], [MeshFrame(1)], [MeshFrame(0)]),
    )
    # -1 (unset) and the empty mesh are skipped; only the real triangle remains.
    assert out.vertices.shape == (3, 3)


def test_combine_empty_model_returns_degenerate_mesh():
    out = combine_model_world([], _model())
    assert out.vertices.shape == (0, 3)
    assert out.faces.shape == (0, 3)
    assert out.materials == []


def test_combine_frame_selection_clamps_to_last():
    rest = _tri_mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    moved = MeshFrame(mesh_index=0, position=np.array([5.0, 0.0, 0.0]))
    placement = [MeshFrame(mesh_index=0), moved]  # frame 0 at origin, frame 1 shifted
    # frame=5 exceeds the placement length and clamps to the last frame.
    out = combine_model_world([rest], Model(meshes=[placement]), frame=5)
    assert np.allclose(out.vertices[0], [5, 0, 0])


def test_face_centroids():
    mesh = _tri_mesh([[0, 0, 0], [3, 0, 0], [0, 3, 0]])
    assert np.allclose(face_centroids(mesh), [[1, 1, 0]])


def test_face_centroids_empty():
    empty = subset_mesh(_tri_mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([False]))
    assert face_centroids(empty).shape == (0, 3)


def test_assign_faces_to_tiles_by_nearest_xz():
    # Two faces, one near tile A (x=0,z=0), one near tile B (x=10,z=0).
    near_a = _tri_mesh([[0, 0, 0], [1, 0, 0], [0, 0, 1]])
    near_b = _tri_mesh([[10, 0, 0], [11, 0, 0], [10, 0, 1]])
    merged = combine_model_world([near_a, near_b], _model([MeshFrame(0)], [MeshFrame(1)]))
    tiles = np.array([[0.0, 0.0], [10.0, 0.0]])
    assignment = assign_faces_to_tiles(merged, tiles)
    assert assignment.tolist() == [0, 1]


def test_assign_uses_xz_not_y():
    # Height (Y) must not influence binning — only horizontal X/Z.
    high = _tri_mesh([[0, 100, 0], [1, 100, 0], [0, 100, 1]])
    tiles = np.array([[0.0, 0.0], [50.0, 0.0]])
    assert assign_faces_to_tiles(high, tiles).tolist() == [0]


def test_assign_faces_to_tiles_empty_mesh():
    empty = Mesh.empty()
    tiles = np.array([[0.0, 0.0]])
    result = assign_faces_to_tiles(empty, tiles)
    assert result.shape == (0,)


def test_subset_mesh_remaps_vertices_tightly():
    a = _tri_mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    b = _tri_mesh([[5, 0, 0], [6, 0, 0], [5, 1, 0]])
    merged = combine_model_world([a, b], _model([MeshFrame(0)], [MeshFrame(1)]))
    sub = subset_mesh(merged, np.array([False, True]))
    # Only the second triangle's 3 vertices survive, reindexed to 0,1,2.
    assert sub.vertices.shape == (3, 3)
    assert np.array_equal(sub.faces, [[0, 1, 2]])
    assert np.allclose(sub.vertices, [[5, 0, 0], [6, 0, 0], [5, 1, 0]])


def test_subset_mesh_empty_mask():
    mesh = _tri_mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    sub = subset_mesh(mesh, np.array([False]))
    assert sub.vertices.shape == (0, 3)
    assert sub.faces.shape == (0, 3)
    # Material list is preserved even when no faces remain.
    assert sub.materials == mesh.materials


def _two_face_mesh(ghost_flags):
    """A 2-triangle mesh; face i uses material i, whose is_ghost = ghost_flags[i]."""
    return Mesh(
        vertices=np.zeros((6, 3), dtype=np.float32),
        normals=np.tile([0.0, 0.0, 1.0], (6, 1)).astype(np.float32),
        uvs=np.zeros((6, 2), dtype=np.float32),
        faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32),
        face_materials=np.array([0, 1], dtype=np.uint32),
        materials=[Material(is_ghost=g) for g in ghost_flags],
    )


def test_split_mesh_by_ghost_no_ghost_is_passthrough():
    mesh = _two_face_mesh([False, False])
    out = split_mesh_by_ghost(mesh, base_mask=2)
    # Uniform-solid meshes are returned as-is (same object) with the base mask.
    assert len(out) == 1
    assert out[0][0] is mesh
    assert out[0][1] == 2


def test_split_mesh_by_ghost_all_ghost_keeps_one_model():
    mesh = _two_face_mesh([True, True])
    out = split_mesh_by_ghost(mesh)
    assert len(out) == 1
    assert out[0][0] is mesh
    assert out[0][1] == int(MeshFlag.GHOST)


def test_split_mesh_by_ghost_mixed_splits_solid_and_ghost():
    mesh = _two_face_mesh([False, True])
    out = split_mesh_by_ghost(mesh, base_mask=4)
    assert len(out) == 2
    (solid, solid_mask), (ghost, ghost_mask) = out
    assert solid_mask == 4
    assert ghost_mask == 4 | int(MeshFlag.GHOST)
    # Each half keeps exactly its one triangle.
    assert solid.faces.shape == (1, 3)
    assert ghost.faces.shape == (1, 3)


def test_split_mesh_by_ghost_empty_mesh():
    empty = Mesh.empty()
    out = split_mesh_by_ghost(empty, base_mask=1)
    assert len(out) == 1
    assert out[0][0] is empty
    assert out[0][1] == 1


def test_split_mesh_by_ghost_material_less_mesh_is_passthrough():
    # A material-less mesh (e.g. an OBJ with no usemtl) has face_materials that
    # index an empty list; it must pass through untouched, not raise.
    mesh = Mesh(
        vertices=np.zeros((3, 3), dtype=np.float32),
        normals=np.tile([0.0, 0.0, 1.0], (3, 1)).astype(np.float32),
        uvs=np.zeros((3, 2), dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.uint32),
        face_materials=np.array([0], dtype=np.uint32),
        materials=[],
    )
    out = split_mesh_by_ghost(mesh, base_mask=3)
    assert len(out) == 1
    assert out[0][0] is mesh
    assert out[0][1] == 3


def _mesh_area(mesh):
    """Sum of triangle areas (for a conservation check across a tile split)."""
    if mesh.faces.shape[0] == 0:
        return 0.0
    tri = mesh.vertices.astype(np.float64)[mesh.faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return float(np.linalg.norm(cross, axis=1).sum() / 2.0)


def test_clip_mesh_to_tile_fully_inside_passthrough():
    mesh = _tri_mesh([[0, 0, 0], [1, 0, 0], [0, 0, 1]])
    out = clip_mesh_to_tile(mesh, (0.0, 0.0), 5.0)
    assert out.faces.shape == (1, 3)
    assert np.allclose(sorted(out.vertices.tolist()), sorted(mesh.vertices.tolist()))


def test_clip_mesh_to_tile_fully_outside_is_empty():
    mesh = _tri_mesh([[100, 0, 100], [101, 0, 100], [100, 0, 101]])
    out = clip_mesh_to_tile(mesh, (0.0, 0.0), 1.0)
    assert out.faces.shape == (0, 3)
    assert out.vertices.shape == (0, 3)
    assert out.materials == mesh.materials


def test_clip_mesh_to_tile_empty_mesh_input():
    out = clip_mesh_to_tile(Mesh.empty(), (0.0, 0.0), 1.0)
    assert out.faces.shape == (0, 3)


def test_clip_mesh_to_tile_clips_straddling_triangle_bounds():
    # Straddles both the X and Z tile boundaries.
    mesh = _tri_mesh([[-2, 5, 0], [2, 5, 0], [0, 5, 2]])
    out = clip_mesh_to_tile(mesh, (0.0, 0.0), 1.0)
    assert out.faces.shape[0] >= 1
    assert np.all(out.vertices[:, 0] >= -1.0 - 1e-5)
    assert np.all(out.vertices[:, 0] <= 1.0 + 1e-5)
    assert np.all(out.vertices[:, 2] >= -1.0 - 1e-5)
    assert np.all(out.vertices[:, 2] <= 1.0 + 1e-5)
    # Height (Y) is untouched by the horizontal clip.
    assert np.allclose(out.vertices[:, 1], 5.0)
    # All resulting triangles keep the source face's material.
    assert np.all(out.face_materials == 0)


def test_clip_mesh_to_tile_interpolates_uv_at_cut():
    # A in-bounds, B in-bounds, C out past x_hi=1: clipping the C-A and B-C
    # edges against the x<=1 plane must linearly interpolate UV too.
    mesh = Mesh(
        vertices=np.array([[0, 0, 0], [0, 0, 1], [3, 0, 0.5]], dtype=np.float32),
        normals=np.tile([0.0, 1.0, 0.0], (3, 1)).astype(np.float32),
        uvs=np.array([[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]], dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.uint32),
        face_materials=np.array([0], dtype=np.uint32),
        materials=[Material()],
    )
    out = clip_mesh_to_tile(mesh, (0.0, 0.0), 1.0)
    cut = out.vertices[:, 0] >= 1.0 - 1e-5
    assert cut.sum() == 2
    # t = (1-3)/(0-3) along C->A, and t = (1-0)/(3-0) along B->C.
    expected_ca = 30.0 + (10.0 - 30.0) * ((1 - 3) / (0 - 3))
    expected_bc = 20.0 + (30.0 - 20.0) * ((1 - 0) / (3 - 0))
    got = sorted(out.uvs[cut, 0].tolist())
    assert np.allclose(got, sorted([expected_ca, expected_bc]), atol=1e-4)


def test_clip_mesh_to_tile_conserves_area_across_tiles():
    # A triangle spanning all 4 quadrants of a 2x2 tile grid (fully contained
    # in the grid's [-1, 1] x [-1, 1] union): the clipped pieces' areas must
    # sum back to the original (no geometry lost or duplicated).
    mesh = _tri_mesh([[-0.9, 0, -0.9], [0.9, 0, -0.7], [-0.7, 0, 0.9]])
    total = _mesh_area(mesh)
    centers = [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)]
    pieces = [clip_mesh_to_tile(mesh, c, 0.5) for c in centers]
    assert np.isclose(sum(_mesh_area(p) for p in pieces), total, rtol=1e-6)
