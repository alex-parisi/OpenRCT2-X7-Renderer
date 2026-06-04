"""
Pure-Python OBJ/MTL loader + Material/Texture types.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from .constants import (
    MATERIAL_BACKGROUND_AA,
    MATERIAL_BACKGROUND_AA_DARK,
    MATERIAL_HAS_TEXTURE,
    MATERIAL_IS_MASK,
    MATERIAL_IS_REMAPPABLE,
    MATERIAL_NO_AO,
    MATERIAL_NO_BLEED,
)
from .palette import _srgb2linear


@dataclass
class Texture:
    width: int
    height: int
    pixels: np.ndarray  # float32 (H, W, 3) linear-RGB


def load_texture(path: Path | str) -> Texture:
    img = PILImage.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    h, w, _ = arr.shape
    linear = _srgb2linear(arr.astype(np.float64) / 255.0).astype(np.float32)
    return Texture(width=w, height=h, pixels=linear)


@dataclass
class Material:
    flags: int = 0
    region: int = 0
    specular_exponent: float = 50.0
    specular_color: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.5, 0.5], dtype=np.float64)
    )
    ambient_color: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64)
    )
    # One of (color, texture) is meaningful, depending on flags.
    color: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5], dtype=np.float64))
    texture: Texture | None = None
    # Python-only classification (not a renderer flag): faces with a glass
    # material are split into the translucent overlay block for wall scenery.
    is_glass: bool = False
    # Double-sided wall face sides: a *Back* face appears only in the rear
    # sprite block, a *Front* face only in the front block; untagged faces (both
    # False) are shared and appear in both. See render_wall.
    is_back: bool = False
    is_front: bool = False


def _classify_material_name(material: Material, name: str) -> None:
    """Apply the same substring-priority rules as Mesh.cpp lines 168-191."""
    if "Remap1" in name:
        material.flags |= MATERIAL_IS_REMAPPABLE
        material.region = 1
    elif "Remap2" in name:
        material.flags |= MATERIAL_IS_REMAPPABLE
        material.region = 2
    elif "Remap3" in name:
        material.flags |= MATERIAL_IS_REMAPPABLE
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

    if "Mask" in name:
        material.flags |= MATERIAL_IS_MASK

    if "NoAO" in name:
        material.flags |= MATERIAL_NO_AO
    if "Edge" in name:
        material.flags |= MATERIAL_BACKGROUND_AA
    if "DarkEdge" in name:
        material.flags |= MATERIAL_BACKGROUND_AA_DARK
    if "NoBleed" in name:
        material.flags |= MATERIAL_NO_BLEED


def _parse_mtl(path: Path, base_dir: Path) -> dict[str, Material]:
    materials: dict[str, Material] = {}
    current: Material | None = None
    current_name = ""
    spec_strength = 1.0
    try:
        text = Path(path).read_text()
    except FileNotFoundError:
        return materials

    def commit():
        if current is not None:
            current.specular_color = current.specular_color * spec_strength
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
            spec_strength = 1.0
        elif current is None:
            continue
        elif cmd == "Kd" and len(parts) >= 4:
            current.color = np.array(
                [float(parts[1]), float(parts[2]), float(parts[3])],
                dtype=np.float64,
            )
        elif cmd == "Ks" and len(parts) >= 4:
            current.specular_color = np.array(
                [float(parts[1]), float(parts[2]), float(parts[3])],
                dtype=np.float64,
            )
        elif cmd == "Ka" and len(parts) >= 4:
            current.ambient_color = np.array(
                [float(parts[1]), float(parts[2]), float(parts[3])],
                dtype=np.float64,
            )
        elif cmd == "Ns" and len(parts) >= 2:
            current.specular_exponent = float(parts[1])
        elif cmd == "map_Kd" and len(parts) >= 2:
            tex_path = " ".join(parts[1:])
            tex_full = (base_dir / tex_path) if not Path(tex_path).is_absolute() else Path(tex_path)
            if not tex_full.exists():
                # Try as given (relative to CWD), matching the assimp behavior.
                tex_full = Path(tex_path)
            try:
                current.texture = load_texture(tex_full)
                current.flags |= MATERIAL_HAS_TEXTURE
            except FileNotFoundError:
                print(f'Failed to load texture "{tex_path}"')
    commit()
    return materials


@dataclass
class Mesh:
    vertices: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    faces: np.ndarray
    face_materials: np.ndarray
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


def _generate_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Area-weighted per-face normals averaged at shared vertices.
    """
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
    return normals / norms[:, None]


def load_mesh(filename: str | Path, transform: np.ndarray | None = None) -> Mesh:
    """
    Load an OBJ + MTL into a Mesh.
    """
    if transform is None:
        transform = np.eye(3, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    det = np.linalg.det(transform)
    flip_winding = det < 0
    if abs(abs(det) - 1.0) > 0.001:
        print("Warning: transformation matrix is not orthonormal")

    path = Path(filename)
    base_dir = path.parent
    text = path.read_text()

    raw_verts: list[tuple[float, float, float]] = []
    raw_norms: list[tuple[float, float, float]] = []
    raw_uvs: list[tuple[float, float]] = []
    # Each face: (mat_index, [(v_idx, vt_idx, vn_idx), ...]).
    raw_faces: list[tuple[int, list[tuple[int, int, int]]]] = []

    materials: dict[str, Material] = {}
    # ordered list of material names in use
    material_order: list[str] = []
    name_to_index: dict[str, int] = {}

    def material_index(name: str) -> int:
        if name not in name_to_index:
            name_to_index[name] = len(material_order)
            material_order.append(name)
        return name_to_index[name]

    current_material = 0
    has_any_face = False

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
            mtl_path = (base_dir / parts[1]) if not Path(parts[1]).is_absolute() else Path(parts[1])
            materials.update(_parse_mtl(mtl_path, base_dir))
        elif cmd == "usemtl" and len(parts) >= 2:
            current_material = material_index(parts[1])
        elif cmd == "f":
            indices = []
            for token in parts[1:]:
                bits = token.split("/")
                v_idx = int(bits[0])
                vt_idx = int(bits[1]) if len(bits) > 1 and bits[1] else 0
                vn_idx = int(bits[2]) if len(bits) > 2 and bits[2] else 0
                indices.append((v_idx, vt_idx, vn_idx))
            if len(indices) < 3:
                continue
            # Fan triangulation.
            for k in range(1, len(indices) - 1):
                tri = [indices[0], indices[k], indices[k + 1]]
                if flip_winding:
                    tri = [tri[0], tri[2], tri[1]]
                raw_faces.append((current_material, tri))
            has_any_face = True

    if not has_any_face:
        # Empty mesh — still produce a valid (degenerate) Mesh.
        return Mesh.empty()

    # Resolve obj indices (1-based, negative = relative).
    nv = len(raw_verts)
    nvn = len(raw_norms)
    nvt = len(raw_uvs)

    def resolve(idx: int, n: int) -> int:
        if idx > 0:
            return idx - 1
        return n + idx

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
            vtr = resolve(vt_idx, nvt) if vt_idx != 0 else -1
            vnr = resolve(vn_idx, nvn) if vn_idx != 0 else -1
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
        normals_f64 = _generate_normals(vertices_f64, faces_u32)

    # Re-normalize normals (transform may scale).
    n_norms = np.linalg.norm(normals_f64, axis=1, keepdims=True)
    n_norms = np.where(n_norms == 0, 1.0, n_norms)
    normals_f64 = normals_f64 / n_norms

    # Assemble material list in the same order as referenced in OBJ.
    mat_list = [materials.get(name, Material()) for name in material_order]

    print(f"Loading model with {vertices_f64.shape[0]} vertices and {faces_u32.shape[0]} faces")

    return Mesh(
        vertices=vertices_f64.astype(np.float32),
        normals=normals_f64.astype(np.float32),
        uvs=uvs_f64.astype(np.float32),
        faces=faces_u32,
        face_materials=face_materials_u32,
        materials=mat_list,
    )
