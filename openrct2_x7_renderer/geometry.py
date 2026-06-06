"""
Mesh geometry utilities for multi-tile (large scenery) rendering: bake a
Model's placements into one world-space mesh, compute face footprints, and
slice a mesh into the subset of faces belonging to a tile.
"""

__all__ = [
    "assign_faces_to_tiles",
    "combine_model_world",
    "face_centroids",
    "rotate_x",
    "rotate_y",
    "rotate_z",
    "subset_mesh",
]

import math

import numpy as np
from numpy.typing import NDArray

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
