"""
Generic config parsing + validation helpers shared by the generators' loaders.

These mirror the small validation helpers the vehicle loader grew; sharing them
keeps the scenery loader from depending on the vehicle package.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from .image import read_png
from .mesh import Mesh, load_mesh
from .types import IndexedImage


class LoadError(Exception):
    """Raised when a config file is malformed or fails validation."""


def parse_config(path: Path | str) -> dict:
    """Parse a JSON or YAML config file into a dict (chosen by extension)."""
    p = Path(path)
    text = p.read_text()
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise LoadError(
                "PyYAML is required to load .yaml configs (pip install pyyaml)"
            ) from None
        root = yaml.safe_load(text)
    else:
        root = json.loads(text)
    if not isinstance(root, dict):
        raise LoadError("Config root is not an object")
    return root


def require_string(obj: dict, key: str) -> str:
    v = obj.get(key)
    if not isinstance(v, str):
        raise LoadError(f'Property "{key}" not found or is not a string')
    return v


def optional_string(obj: dict, key: str, default: str = "") -> str:
    v = obj.get(key)
    if v is None:
        return default
    if not isinstance(v, str):
        raise LoadError(f'Property "{key}" is not a string')
    return v


def optional_string_list(obj: dict, key: str) -> list[str]:
    v = obj.get(key)
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if not isinstance(v, list) or any(not isinstance(x, str) for x in v):
        raise LoadError(f'Property "{key}" is not a string or array of strings')
    return list(v)


def require_int(obj: dict, key: str) -> int:
    v = obj.get(key)
    if not isinstance(v, int) or isinstance(v, bool):
        raise LoadError(f'Property "{key}" not found or is not an integer')
    return v


def optional_int(obj: dict, key: str, default: int) -> int:
    v = obj.get(key)
    if v is None:
        return default
    if not isinstance(v, int) or isinstance(v, bool):
        raise LoadError(f'Property "{key}" is not an integer')
    return v


def require_number(obj: dict, key: str) -> float:
    v = obj.get(key)
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        raise LoadError(f'Property "{key}" not found or is not a number')
    return float(v)


def optional_number(obj: dict, key: str, default: float) -> float:
    v = obj.get(key)
    if v is None:
        return default
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        raise LoadError(f'Property "{key}" is not a number')
    return float(v)


def optional_bool(obj: dict, key: str, default: bool = False) -> bool:
    v = obj.get(key)
    if v is None:
        return default
    if not isinstance(v, bool):
        raise LoadError(f'Property "{key}" is not a boolean')
    return v


def read_vector3(arr: Any) -> np.ndarray:
    if not isinstance(arr, list) or len(arr) != 3:
        raise LoadError("Vector must be an array of 3 numbers")
    return np.array([float(x) for x in arr], dtype=np.float64)


def as_array_or_wrap(value: Any) -> list:
    if value is None:
        raise LoadError("Missing value")
    if isinstance(value, list):
        if len(value) == 0:
            raise LoadError("Empty array")
        return value
    return [value]


def load_meshes(root: dict) -> list[Mesh]:
    """Load every OBJ listed under the config's `meshes` array."""
    mesh_paths = root.get("meshes")
    if not isinstance(mesh_paths, list):
        raise LoadError('Property "meshes" does not exist or is not an array')
    meshes: list[Mesh] = []
    for path in mesh_paths:
        if not isinstance(path, str):
            raise LoadError("Mesh path is not a string")
        meshes.append(load_mesh(path))
    return meshes


def load_preview(root: dict) -> IndexedImage | None:
    """Load the optional `preview` PNG referenced by the config, if any."""
    preview_path = root.get("preview")
    if preview_path is None:
        return None
    if not isinstance(preview_path, str):
        raise LoadError('Property "preview" is not a string')
    try:
        return read_png(preview_path)
    except Exception as e:
        raise LoadError(f"Unable to open image file {preview_path}: {e}") from e
