"""Tests for the multi-tile (large scenery) geometry baking utilities."""

import numpy as np
from openrct2_x7_renderer.geometry import (
    assign_faces_to_tiles,
    combine_model_world,
    face_centroids,
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
