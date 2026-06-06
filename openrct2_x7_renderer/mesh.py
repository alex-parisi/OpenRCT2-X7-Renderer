"""
Pure-Python OBJ/MTL loader + Material/Texture types.
"""

__all__ = ["Material", "Mesh", "Texture", "load_mesh", "load_texture"]

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage

from .constants import MaterialFlag
from .palette import srgb2linear
from .types import LoadError

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, eq=False)
class Texture:
    """Linear-RGB texture loaded from an image file.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        pixels: float32 ``(H, W, 3)`` array of linear-RGB values in ``[0, 1]``.
    """

    width: int
    height: int
    pixels: NDArray[np.float32]  # (H, W, 3) linear-RGB


def load_texture(path: Path | str) -> Texture:
    """Load an image file and return it as a linear-RGB float32 Texture."""
    img = PILImage.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    h, w, _ = arr.shape
    linear = srgb2linear(arr.astype(np.float64) / 255.0).astype(np.float32)
    return Texture(width=w, height=h, pixels=linear)


@dataclass
class Material:
    """OBJ material with flags, diffuse color or texture, and Phong shading parameters.

    Attributes:
        flags: Bitmask of ``MaterialFlag`` values from ``constants.py``.
        region: Palette remap region (0 = none, 1–3 = remappable, 4 = greyscale, 5 = peep).
        specular_exponent: Phong shininess exponent (``Ns``).
        specular_color: ``(3,)`` linear-RGB specular reflectance (``Ks``).
        ambient_color: ``(3,)`` linear-RGB ambient reflectance (``Ka``).
        color: ``(3,)`` linear-RGB diffuse color; used when ``MaterialFlag.HAS_TEXTURE`` is unset.
        texture: Loaded diffuse texture; present when ``MaterialFlag.HAS_TEXTURE`` is set.
        is_glass: True for glass materials rendered in a separate translucent pass.
        is_back: True for faces that appear only in the rear wall sprite block.
        is_front: True for faces that appear only in the front wall sprite block.
    """

    flags: int = 0
    region: int = 0
    specular_exponent: float = 50.0
    specular_color: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.5, 0.5, 0.5], dtype=np.float64)
    )
    ambient_color: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64)
    )
    color: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.5, 0.5, 0.5], dtype=np.float64)
    )
    texture: Texture | None = None
    # Wall-specific flags:
    is_glass: bool = False
    is_back: bool = False
    is_front: bool = False


def _classify_material_name(material: Material, name: str) -> None:
    """Set material flags by scanning for known substrings in the OBJ material name."""
    if "Remap1" in name:
        material.flags |= MaterialFlag.IS_REMAPPABLE
        material.region = 1
    elif "Remap2" in name:
        material.flags |= MaterialFlag.IS_REMAPPABLE
        material.region = 2
    elif "Remap3" in name:
        material.flags |= MaterialFlag.IS_REMAPPABLE
        material.region = 3
    elif "Greyscale" in name:
        material.region = 4
    elif "Peep" in name:
        material.region = 5

    if "Glass" in name:
        material.is_glass = True

    if "Back" in name:
        material.is_back = True
    elif "Front" in name:
        material.is_front = True

    # "VisibleMask" must be tested before "Mask" (it contains the substring): a visible
    # mask is rendered into the silhouette/occlusion pass, whereas a plain mask is an
    # invisible occluder. Mirrors IsoRender's model.cpp material-flag parsing.
    if "VisibleMask" in name:
        material.flags |= MaterialFlag.IS_VISIBLE_MASK
    elif "Mask" in name:
        material.flags |= MaterialFlag.IS_MASK

    if "NoAO" in name:
        material.flags |= MaterialFlag.NO_AO
    if "DarkEdge" in name:
        material.flags |= MaterialFlag.BACKGROUND_AA_DARK
    elif "Edge" in name:
        material.flags |= MaterialFlag.BACKGROUND_AA
    if "NoBleed" in name:
        material.flags |= MaterialFlag.NO_BLEED


def _parse_mtl(path: Path, base_dir: Path) -> dict[str, Material]:
    """Parse a ``.mtl`` file and return a mapping from material name to Material.

    ``Kd``, ``Ks``, and ``Ka`` colour values are treated as sRGB and converted
    to linear RGB, matching the conversion applied to ``map_Kd`` textures.

    Missing files are logged as a warning and treated as empty.
    Textures that cannot be found are skipped with a logged warning.
    """
    materials: dict[str, Material] = {}
    current: Material | None = None
    current_name = ""
    try:
        text = Path(path).read_text()
    except FileNotFoundError:
        _log.warning('MTL file not found: "%s"', path)
        return materials

    def commit() -> None:
        if current is not None:
            _classify_material_name(current, current_name)
            materials[current_name] = current

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cmd = parts[0]
        if cmd == "newmtl":
            commit()
            current = Material()
            current_name = parts[1] if len(parts) > 1 else ""
        elif current is None:
            continue
        elif cmd == "Kd" and len(parts) >= 4:
            current.color = srgb2linear(np.array(
                [float(parts[1]), float(parts[2]), float(parts[3])],
                dtype=np.float64,
            ))
        elif cmd == "Ks" and len(parts) >= 4:
            current.specular_color = srgb2linear(np.array(
                [float(parts[1]), float(parts[2]), float(parts[3])],
                dtype=np.float64,
            ))
        elif cmd == "Ka" and len(parts) >= 4:
            current.ambient_color = srgb2linear(np.array(
                [float(parts[1]), float(parts[2]), float(parts[3])],
                dtype=np.float64,
            ))
        elif cmd == "Ns" and len(parts) >= 2:
            current.specular_exponent = float(parts[1])
        elif cmd == "map_Kd" and len(parts) >= 2:
            tex_path = " ".join(parts[1:])
            tex_full = (base_dir / tex_path) if not Path(tex_path).is_absolute() else Path(tex_path)
            if not tex_full.exists():
                tex_full = Path(tex_path)
            try:
                current.texture = load_texture(tex_full)
                current.flags |= MaterialFlag.HAS_TEXTURE
            except FileNotFoundError:
                _log.warning('Failed to load texture "%s"', tex_path)
    commit()
    return materials


@dataclass
class Mesh:
    """Triangle mesh with per-vertex geometry attributes and a material list.

    Attributes:
        vertices: float32 ``(V, 3)`` vertex positions.
        normals: float32 ``(V, 3)`` per-vertex normals (unit length).
        uvs: float32 ``(V, 2)`` texture coordinates.
        faces: uint32 ``(F, 3)`` triangle indices into the vertex arrays.
        face_materials: uint32 ``(F,)`` material index per face.
        materials: Material list; indices correspond to ``face_materials`` values.
    """

    vertices: NDArray[np.float32]
    normals: NDArray[np.float32]
    uvs: NDArray[np.float32]
    faces: NDArray[np.uint32]
    face_materials: NDArray[np.uint32]
    materials: list[Material]

    @classmethod
    def empty(cls, materials: list[Material] | None = None) -> "Mesh":
        """A valid mesh with no geometry (used for degenerate/empty cases)."""
        return cls(
            vertices=np.zeros((0, 3), dtype=np.float32),
            normals=np.zeros((0, 3), dtype=np.float32),
            uvs=np.zeros((0, 2), dtype=np.float32),
            faces=np.zeros((0, 3), dtype=np.uint32),
            face_materials=np.zeros((0,), dtype=np.uint32),
            materials=materials if materials is not None else [],
        )


def _generate_normals(
    vertices: NDArray[np.float64], faces: NDArray[np.uint32]
) -> NDArray[np.float64]:
    """Area-weighted per-face normals accumulated at shared vertices, normalized to unit length."""
    normals = np.zeros_like(vertices)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)  # area-weighted
    np.add.at(normals, faces[:, 0], face_normals)
    np.add.at(normals, faces[:, 1], face_normals)
    np.add.at(normals, faces[:, 2], face_normals)
    norms = np.linalg.norm(normals, axis=1)
    norms[norms == 0] = 1.0
    result: NDArray[np.float64] = normals / norms[:, None]
    return result


def load_mesh(filename: str | Path, transform: NDArray[np.float64] | None = None) -> Mesh:
    """Load an OBJ file (and the MTL libraries it references) into a Mesh.

    The OBJ is parsed into a vertex-deduplicated triangle mesh.  Polygons are
    fan-triangulated.  Multiple ``mtllib`` filenames on one line are all loaded.

    Normal handling: if the file contains ``vn`` entries *and* every face vertex
    references a normal, the artist normals are used and transformed.  If normals
    are defined but some face vertices omit the ``//vn`` token, a WARNING is
    logged and area-weighted face normals are generated for the entire mesh
    instead.  If no ``vn`` entries exist at all, face normals are generated
    silently.

    UV handling: if a face vertex references a texture coordinate index but no
    ``vt`` entries are defined in the file, a WARNING is logged and the UV for
    that vertex defaults to ``(0, 0)``.

    Args:
        filename: Path to the ``.obj`` file.
        transform: Optional 3×3 orthonormal matrix applied to vertex positions
            and normals at load time (e.g. axis-swap or mirror).  A negative
            determinant flips face winding order.  Defaults to the identity.
            Raises ``LoadError`` if ``|det| − 1 > 0.001`` (not orthonormal).

    Raises:
        LoadError: If the file does not exist, the transform is not orthonormal,
            a face index is out of range, or a face token cannot be parsed.
    """
    if transform is None:
        transform = np.eye(3, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    det = np.linalg.det(transform)
    flip_winding = det < 0
    if abs(abs(det) - 1.0) > 0.001:
        raise LoadError(
            f"transformation matrix is not orthonormal "
            f"(|det| = {abs(det):.6f}, expected 1.0)"
        )

    path = Path(filename)
    base_dir = path.parent
    try:
        text = path.read_text()
    except FileNotFoundError:
        raise LoadError(f"Mesh file not found: {path}") from None

    raw_verts: list[tuple[float, float, float]] = []
    raw_norms: list[tuple[float, float, float]] = []
    raw_uvs: list[tuple[float, float]] = []
    # Each face: (mat_index, [(v_idx, vt_idx, vn_idx), ...]).
    raw_faces: list[tuple[int, list[tuple[int, int, int]]]] = []

    materials: dict[str, Material] = {}
    material_order: list[str] = []
    name_to_index: dict[str, int] = {}

    def material_index(name: str) -> int:
        if name not in name_to_index:
            name_to_index[name] = len(material_order)
            material_order.append(name)
        return name_to_index[name]

    current_material = 0

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cmd = parts[0]
        if cmd == "v" and len(parts) >= 4:
            raw_verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif cmd == "vn" and len(parts) >= 4:
            raw_norms.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif cmd == "vt" and len(parts) >= 3:
            raw_uvs.append((float(parts[1]), float(parts[2])))
        elif cmd == "mtllib" and len(parts) >= 2:
            for mtl_name in parts[1:]:
                name_path = Path(mtl_name)
                mtl_path = name_path if name_path.is_absolute() else (base_dir / mtl_name)
                materials.update(_parse_mtl(mtl_path, base_dir))
        elif cmd == "usemtl" and len(parts) >= 2:
            current_material = material_index(parts[1])
        elif cmd == "f":
            indices = []
            try:
                for token in parts[1:]:
                    bits = token.split("/")
                    v_idx = int(bits[0])
                    vt_idx = int(bits[1]) if len(bits) > 1 and bits[1] else 0
                    vn_idx = int(bits[2]) if len(bits) > 2 and bits[2] else 0
                    indices.append((v_idx, vt_idx, vn_idx))
            except ValueError as e:
                raise LoadError(f"Malformed face index in {path}: {e}") from None
            if len(indices) < 3:
                _log.warning("Skipping degenerate face with fewer than 3 vertices in %s", path)
                continue
            # Fan triangulation.
            for k in range(1, len(indices) - 1):
                tri = [indices[0], indices[k], indices[k + 1]]
                if flip_winding:
                    tri = [tri[0], tri[2], tri[1]]
                raw_faces.append((current_material, tri))

    if not raw_faces:
        return Mesh.empty()

    # Resolve obj indices
    nv = len(raw_verts)
    nvn = len(raw_norms)
    nvt = len(raw_uvs)

    def resolve(idx: int, n: int) -> int:
        result = (idx - 1) if idx > 0 else (n + idx)
        if result < 0 or result >= n:
            raise LoadError(
                f"OBJ index {idx} is out of range for {n} elements in {path}"
            )
        return result

    # Build expanded vertex list keyed by (v, vt, vn) tuples
    vert_cache: dict[tuple[int, int, int], int] = {}
    out_vertices: list[tuple[float, float, float]] = []
    out_normals_raw: list[tuple[float, float, float] | None] = []
    out_uvs: list[tuple[float, float]] = []
    out_faces: list[tuple[int, int, int]] = []
    out_face_materials: list[int] = []

    has_any_normal_input = nvn > 0

    for mat_idx, tri in raw_faces:
        face_idx = []
        for v_idx, vt_idx, vn_idx in tri:
            vr = resolve(v_idx, nv)
            if vt_idx != 0 and nvt == 0:
                _log.warning(
                    "Face in %s references texture coordinate %d but no vt entries are defined"
                    " — UV will default to (0, 0)",
                    path,
                    vt_idx,
                )
            vtr = resolve(vt_idx, nvt) if vt_idx != 0 and nvt > 0 else -1
            vnr = resolve(vn_idx, nvn) if vn_idx != 0 and nvn > 0 else -1
            key = (vr, vtr, vnr)
            if key not in vert_cache:
                vert_cache[key] = len(out_vertices)
                out_vertices.append(raw_verts[vr])
                out_normals_raw.append(raw_norms[vnr] if vnr >= 0 else None)
                out_uvs.append(raw_uvs[vtr] if vtr >= 0 else (0.0, 0.0))
            face_idx.append(vert_cache[key])
        out_faces.append((face_idx[0], face_idx[1], face_idx[2]))
        out_face_materials.append(mat_idx)

    # Keep float64 throughout geometry processing for precision
    vertices_f64 = np.array(out_vertices, dtype=np.float64)
    uvs_f64 = np.array(out_uvs, dtype=np.float64)
    faces_u32 = np.array(out_faces, dtype=np.uint32)
    face_materials_u32 = np.array(out_face_materials, dtype=np.uint32)

    # Apply transform.
    vertices_f64 = vertices_f64 @ transform.T

    if has_any_normal_input and all(n is not None for n in out_normals_raw):
        normals_f64 = np.array(out_normals_raw, dtype=np.float64)
        normals_f64 = normals_f64 @ transform.T
    else:
        if has_any_normal_input:
            _log.warning(
                "%s defines vertex normals but some face vertices omit normal references"
                " — discarding all vn data and generating normals from geometry",
                path,
            )
        normals_f64 = _generate_normals(vertices_f64, faces_u32)

    # Re-normalize normals
    n_norms = np.linalg.norm(normals_f64, axis=1, keepdims=True)
    n_norms = np.where(n_norms == 0, 1.0, n_norms)
    normals_f64 = normals_f64 / n_norms

    # Assemble material list in the same order as referenced in OBJ.
    mat_list = [materials.get(name, Material()) for name in material_order]

    _log.debug("Loading model: %d vertices, %d faces", vertices_f64.shape[0], faces_u32.shape[0])

    return Mesh(
        vertices=vertices_f64.astype(np.float32),
        normals=normals_f64.astype(np.float32),
        uvs=uvs_f64.astype(np.float32),
        faces=faces_u32,
        face_materials=face_materials_u32,
        materials=mat_list,
    )
