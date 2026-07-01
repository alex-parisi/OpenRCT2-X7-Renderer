"""
Mesh geometry utilities for multi-tile (large scenery) rendering: bake a
Model's placements into one world-space mesh, compute face footprints, and
slice a mesh into the subset of faces belonging to a tile.
"""

__all__ = [
    "assign_faces_to_tiles",
    "clip_mesh_to_tile",
    "combine_model_world",
    "face_centroids",
    "rotate_x",
    "rotate_y",
    "rotate_z",
    "split_mesh_by_ghost",
    "subset_mesh",
]

import math

import numpy as np
from numpy.typing import NDArray

from .constants import MeshFlag
from .mesh import Material, Mesh
from .types import Model


def rotate_x(theta: float) -> NDArray[np.float64]:
    """Return the 3x3 rotation matrix about the X axis by ``theta`` radians."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rotate_y(theta: float) -> NDArray[np.float64]:
    """Return the 3x3 rotation matrix about the Y axis by ``theta`` radians."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rotate_z(theta: float) -> NDArray[np.float64]:
    """Return the 3x3 rotation matrix about the Z axis by ``theta`` radians."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def combine_model_world(meshes: list[Mesh], model: Model, frame: int = 0) -> Mesh:
    """Bake a Model's placements into a single world-space Mesh.

    Applies each placement's rotation and translation, then concatenates
    geometry and merges material lists across all placements.

    Args:
        meshes: Source mesh list indexed by ``MeshFrame.mesh_index``.
        model: Animated object describing the placement hierarchy.
        frame: Animation frame index to bake (default 0). Placements with
            fewer frames than ``frame`` fall back to their last frame.

    Returns:
        A new Mesh in world space, or an empty Mesh if no placements have
        geometry.
    """
    vs: list[NDArray[np.float32]] = []
    ns: list[NDArray[np.float32]] = []
    uvs: list[NDArray[np.float32]] = []
    fs: list[NDArray[np.uint32]] = []
    fms: list[NDArray[np.uint32]] = []
    materials: list[Material] = []
    v_off = 0
    m_off = 0
    for placement in model.meshes:
        mf = placement[min(frame, len(placement) - 1)]
        if mf.mesh_index == -1:
            continue
        mesh = meshes[mf.mesh_index]
        if mesh.faces.shape[0] == 0:
            continue
        angle_y, angle_z, angle_x = mf.orientation * math.pi / 180.0
        rot = rotate_y(angle_y) @ rotate_z(angle_z) @ rotate_x(angle_x)
        t = mf.position.astype(np.float64)

        v = mesh.vertices.astype(np.float64) @ rot.T + t
        n = mesh.normals.astype(np.float64) @ rot.T
        norms = np.linalg.norm(n, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        n = n / norms

        vs.append(v.astype(np.float32))
        ns.append(n.astype(np.float32))
        uvs.append(mesh.uvs.astype(np.float32))
        fs.append(mesh.faces.astype(np.uint32) + np.uint32(v_off))
        fms.append(mesh.face_materials.astype(np.uint32) + np.uint32(m_off))
        materials.extend(mesh.materials)
        v_off += mesh.vertices.shape[0]
        m_off += len(mesh.materials)

    if not vs:
        return Mesh.empty()

    return Mesh(
        vertices=np.concatenate(vs, axis=0),
        normals=np.concatenate(ns, axis=0),
        uvs=np.concatenate(uvs, axis=0),
        faces=np.concatenate(fs, axis=0),
        face_materials=np.concatenate(fms, axis=0),
        materials=materials,
    )


def face_centroids(mesh: Mesh) -> NDArray[np.float64]:
    """(F, 3) centroid of each triangle face."""
    if mesh.faces.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    tri = mesh.vertices.astype(np.float64)[mesh.faces]  # (F, 3, 3)
    centroids: NDArray[np.float64] = tri.mean(axis=1)
    return centroids


def assign_faces_to_tiles(mesh: Mesh, tile_centers_xz: NDArray[np.float64]) -> NDArray[np.intp]:
    """Return a (F,) array assigning each face to the nearest tile by horizontal
    (OBJ X, Z) distance. `tile_centers_xz` is (T, 2) in OBJ units."""
    cents = face_centroids(mesh)
    if cents.shape[0] == 0:
        return np.zeros((0,), dtype=np.intp)
    xz = cents[:, (0, 2)]  # OBJ X and Z are the horizontal plane (+Y is up)
    # (F, T) squared distances.
    d = ((xz[:, None, :] - tile_centers_xz[None, :, :]) ** 2).sum(axis=2)
    nearest: NDArray[np.intp] = np.argmin(d, axis=1)
    return nearest


def subset_mesh(mesh: Mesh, face_mask: NDArray[np.bool_]) -> Mesh:
    """Build a Mesh from the selected faces, remapping to only referenced
    vertices so scene bounds stay tight per tile."""
    faces = mesh.faces[face_mask]
    if faces.shape[0] == 0:
        return Mesh.empty(mesh.materials)
    used = np.unique(faces.reshape(-1))
    remap = np.full(mesh.vertices.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0])
    new_faces = remap[faces].astype(np.uint32)
    return Mesh(
        vertices=mesh.vertices[used],
        normals=mesh.normals[used],
        uvs=mesh.uvs[used],
        faces=new_faces,
        face_materials=mesh.face_materials[face_mask].astype(np.uint32),
        materials=mesh.materials,
    )


def _clip_polygon_axis(
    poly: list[NDArray[np.float64]], axis: int, bound: float, *, keep_less: bool
) -> list[NDArray[np.float64]]:
    """Sutherland-Hodgman clip of a convex polygon against one axis-aligned half-plane.

    Each polygon vertex is a full attribute row (position, normal, uv
    concatenated) so the cut edge's normal/uv are linearly interpolated along
    with position. Keeps ``coord <= bound`` when `keep_less`, else ``coord >=
    bound``.
    """
    if not poly:
        return poly
    out: list[NDArray[np.float64]] = []
    n = len(poly)
    for i in range(n):
        cur = poly[i]
        prev = poly[i - 1]
        cur_val, prev_val = cur[axis], prev[axis]
        cur_in = cur_val <= bound if keep_less else cur_val >= bound
        prev_in = prev_val <= bound if keep_less else prev_val >= bound
        if cur_in != prev_in:
            t = (bound - prev_val) / (cur_val - prev_val)
            out.append(prev + (cur - prev) * t)
        if cur_in:
            out.append(cur)
    return out


def clip_mesh_to_tile(
    mesh: Mesh, center_xz: tuple[float, float], half_size: float
) -> Mesh:
    """The portion of `mesh` inside one axis-aligned tile footprint (OBJ X/Z
    square of side ``2 * half_size`` centred on `center_xz`), clipping
    triangles that straddle the boundary instead of discarding or keeping them
    whole.

    Large-scenery tiles need one sprite per tile showing only the geometry
    over that tile's footprint. Assigning each *whole* triangle to its nearest
    tile (see :func:`assign_faces_to_tiles`) keeps a straddling triangle's far
    vertices verbatim: for continuous geometry spanning several tiles (a
    cylindrical tank, a conical roof's apex fan), that leaves slivers reaching
    clear across the model, wildly inflating the tile's render bounds. This
    clips each triangle against the tile's square instead, fan-triangulating
    the resulting convex polygon (at most a heptagon, from 3 vertices clipped
    by 4 half-planes) and linearly interpolating normal/uv at new cut edges.
    """
    cx, cz = center_xz
    x_lo, x_hi = cx - half_size, cx + half_size
    z_lo, z_hi = cz - half_size, cz + half_size

    verts = mesh.vertices.astype(np.float64)
    norms = mesh.normals.astype(np.float64)
    uvs = mesh.uvs.astype(np.float64)

    out_rows: list[NDArray[np.float64]] = []
    out_faces: list[tuple[int, int, int]] = []
    out_face_materials: list[int] = []

    for f in range(mesh.faces.shape[0]):
        poly = [np.concatenate([verts[i], norms[i], uvs[i]]) for i in mesh.faces[f]]
        poly = _clip_polygon_axis(poly, 0, x_lo, keep_less=False)
        poly = _clip_polygon_axis(poly, 0, x_hi, keep_less=True)
        poly = _clip_polygon_axis(poly, 2, z_lo, keep_less=False)
        poly = _clip_polygon_axis(poly, 2, z_hi, keep_less=True)
        if len(poly) < 3:
            continue
        base = len(out_rows)
        out_rows.extend(poly)
        material = int(mesh.face_materials[f])
        for i in range(1, len(poly) - 1):
            out_faces.append((base, base + i, base + i + 1))
            out_face_materials.append(material)

    if not out_faces:
        return Mesh.empty(mesh.materials)

    rows = np.array(out_rows, dtype=np.float64)
    n = rows[:, 3:6]
    n_len = np.linalg.norm(n, axis=1, keepdims=True)
    n_len[n_len == 0] = 1.0

    return Mesh(
        vertices=rows[:, 0:3].astype(np.float32),
        normals=(n / n_len).astype(np.float32),
        uvs=rows[:, 6:8].astype(np.float32),
        faces=np.array(out_faces, dtype=np.uint32),
        face_materials=np.array(out_face_materials, dtype=np.uint32),
        materials=mesh.materials,
    )


def split_mesh_by_ghost(mesh: Mesh, base_mask: int = 0) -> list[tuple[Mesh, int]]:
    """Split a mesh into ``(sub-mesh, MeshFlag mask)`` pairs by ``Material.is_ghost``.

    Faces whose material is a ghost get ``base_mask | MeshFlag.GHOST`` so the
    renderer traces through them (silhouette/occlusion only); the rest keep
    ``base_mask``. Returns a single pair when the mesh is empty or uniform (wholly
    ghost or wholly solid), so non-ghost callers are unaffected.
    """
    n = int(mesh.faces.shape[0])
    # No faces, or a material-less mesh (e.g. an OBJ with no ``usemtl``, whose
    # face_materials index an empty list): nothing can be ghost.
    if n == 0 or not mesh.materials:
        return [(mesh, base_mask)]
    ghost = np.fromiter(
        (mesh.materials[m].is_ghost for m in mesh.face_materials),
        dtype=bool,
        count=n,
    )
    if not ghost.any():
        return [(mesh, base_mask)]
    ghost_mask = base_mask | int(MeshFlag.GHOST)
    if ghost.all():
        return [(mesh, ghost_mask)]
    return [
        (subset_mesh(mesh, ~ghost), base_mask),
        (subset_mesh(mesh, ghost), ghost_mask),
    ]
