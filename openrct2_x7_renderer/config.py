"""
Generic config parsing + validation helpers shared by the generators' loaders.
"""

__all__ = [
    "LoadError",
    "as_array_or_wrap",
    "load_meshes",
    "load_preview",
    "optional_bool",
    "optional_int",
    "optional_number",
    "optional_string",
    "optional_string_list",
    "parse_config",
    "read_vector3",
    "require_int",
    "require_number",
    "require_string",
]

import json
from pathlib import Path
from typing import Any

import numpy as np

from .image import read_png
from .mesh import Mesh, load_mesh
from .types import IndexedImage, LoadError


def parse_config(path: Path | str) -> dict[str, Any]:
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


def require_string(obj: dict[str, Any], key: str) -> str:
    """Return the string at ``obj[key]``, raising LoadError if absent or not a string."""
    v = obj.get(key)
    if not isinstance(v, str):
        raise LoadError(f'Property "{key}" not found or is not a string')
    return v


def optional_string(obj: dict[str, Any], key: str, default: str = "") -> str:
    """Return the string at ``obj[key]``, or ``default`` if the key is absent."""
    v = obj.get(key)
    if v is None:
        return default
    if not isinstance(v, str):
        raise LoadError(f'Property "{key}" is not a string')
    return v


def optional_string_list(obj: dict[str, Any], key: str) -> list[str]:
    """Return the value at ``obj[key]`` coerced to a list of strings.

    A single string is wrapped in a list; an absent key returns ``[]``.
    """
    v = obj.get(key)
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if not isinstance(v, list) or any(not isinstance(x, str) for x in v):
        raise LoadError(f'Property "{key}" is not a string or array of strings')
    return list(v)


def require_int(obj: dict[str, Any], key: str) -> int:
    """Return the integer at ``obj[key]``, raising LoadError if absent, non-int, or a bool."""
    v = obj.get(key)
    if not isinstance(v, int) or isinstance(v, bool):
        raise LoadError(f'Property "{key}" not found or is not an integer')
    return v


def optional_int(obj: dict[str, Any], key: str, default: int) -> int:
    """Return the integer at ``obj[key]``, or ``default`` if the key is absent."""
    v = obj.get(key)
    if v is None:
        return default
    if not isinstance(v, int) or isinstance(v, bool):
        raise LoadError(f'Property "{key}" is not an integer')
    return v


def require_number(obj: dict[str, Any], key: str) -> float:
    """Return the number at ``obj[key]`` as float, raising LoadError if absent or non-numeric."""
    v = obj.get(key)
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        raise LoadError(f'Property "{key}" not found or is not a number')
    return float(v)


def optional_number(obj: dict[str, Any], key: str, default: float) -> float:
    """Return the number at ``obj[key]`` as float, or ``default`` if the key is absent."""
    v = obj.get(key)
    if v is None:
        return default
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        raise LoadError(f'Property "{key}" is not a number')
    return float(v)


def optional_bool(obj: dict[str, Any], key: str, default: bool = False) -> bool:
    """Return the boolean at ``obj[key]``, or ``default`` if the key is absent."""
    v = obj.get(key)
    if v is None:
        return default
    if not isinstance(v, bool):
        raise LoadError(f'Property "{key}" is not a boolean')
    return v


def read_vector3(arr: Any) -> np.ndarray:
    """Parse a 3-element list into a float64 ``(3,)`` array, raising LoadError on bad input."""
    if not isinstance(arr, list) or len(arr) != 3:
        raise LoadError("Vector must be an array of 3 numbers")
    try:
        return np.array([float(x) for x in arr], dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise LoadError(f"Vector element is not a number: {e}") from e


def as_array_or_wrap(value: Any) -> list[Any]:
    """Return ``value`` as-is if it is a non-empty list, or wrap a scalar in a one-element list."""
    if value is None:
        raise LoadError("Missing value")
    if isinstance(value, list):
        if len(value) == 0:
            raise LoadError("Empty array")
        return value
    return [value]


def load_meshes(root: dict[str, Any], base_dir: Path | None = None) -> list[Mesh]:
    """Load every OBJ listed under the config's `meshes` array.

    Relative mesh paths are resolved against *base_dir* (typically the
    directory containing the config file).  When *base_dir* is ``None``
    the paths are used as-is (resolved against CWD).
    """
    mesh_paths = root.get("meshes")
    if not isinstance(mesh_paths, list):
        raise LoadError('Property "meshes" does not exist or is not an array')
    meshes: list[Mesh] = []
    for path in mesh_paths:
        if not isinstance(path, str):
            raise LoadError("Mesh path is not a string")
        resolved = Path(path) if base_dir is None or Path(path).is_absolute() else base_dir / path
        meshes.append(load_mesh(resolved))
    return meshes


def load_preview(root: dict[str, Any], base_dir: Path | None = None) -> IndexedImage | None:
    """Load the optional `preview` PNG referenced by the config, if any.

    Relative preview paths are resolved against *base_dir*.
    """
    preview_path = root.get("preview")
    if preview_path is None:
        return None
    if not isinstance(preview_path, str):
        raise LoadError('Property "preview" is not a string')
    resolved = (
        Path(preview_path)
        if base_dir is None or Path(preview_path).is_absolute()
        else base_dir / preview_path
    )
    try:
        return read_png(resolved)
    except (OSError, ValueError) as e:
        raise LoadError(f"Unable to open image file {preview_path}: {e}") from e
