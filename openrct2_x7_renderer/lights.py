"""
Default light rig + light-config loading, shared by the generators.
Ported from X7's rendering engine
https://github.com/X123M3-256/RCTGen
"""

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import read_vector3, require_number, require_string
from .constants import LightType
from .types import Light, LoadError

_log = logging.getLogger(__name__)


def _normalize(v: list[float] | NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the unit vector of ``v``.

    Raises:
        ValueError: If ``v`` is the zero vector, which would produce an
            undefined light direction.
    """
    arr = np.array(v, dtype=np.float64)
    n = np.linalg.norm(arr)
    if n == 0:
        raise ValueError(f"Light direction cannot be the zero vector: {v}")
    return arr / n


def default_lights() -> list[Light]:
    """Return the default nine-light rig used when no ``lights`` block is in the config."""
    return [
        Light(LightType.DIFFUSE, False, _normalize([0.0, -1.0, 0.0]), 0.1),
        Light(LightType.DIFFUSE, False, _normalize([0.0, 0.5, -1.0]), 0.8),
        Light(LightType.SPECULAR, True, _normalize([1.0, 1.65, -1.0]), 0.5),
        Light(LightType.DIFFUSE, True, _normalize([1.0, 1.7, -1.0]), 0.8),
        Light(LightType.DIFFUSE, False, _normalize([0.0, 1.0, 0.0]), 0.45),
        Light(LightType.DIFFUSE, False, _normalize([-1.0, 0.85, 1.0]), 0.475),
        Light(LightType.DIFFUSE, False, _normalize([0.75, 0.4, -1.0]), 0.6),
        Light(LightType.DIFFUSE, False, _normalize([1.0, 0.25, 0.0]), 0.5),
        Light(LightType.DIFFUSE, False, _normalize([-1.0, -0.5, 0.0]), 0.1),
    ]


def load_lights(value: Any) -> list[Light]:
    """Parse a list of light config dicts into Light objects.

    Args:
        value: The value of the ``lights`` key from the config root dict.

    Returns:
        A list of Light objects; non-object elements are skipped with a warning.

    Raises:
        LoadError: If ``value`` is not a list, a light has an unrecognized ``type``, or a
            required field is missing.
    """
    if not isinstance(value, list):
        raise LoadError('"lights" is not an array')
    out: list[Light] = []
    for light in value:
        if not isinstance(light, dict):
            _log.warning("Light array contains a non-object element — ignoring")
            continue
        type_str = require_string(light, "type")
        if type_str == "diffuse":
            type_val = LightType.DIFFUSE
        elif type_str == "specular":
            type_val = LightType.SPECULAR
        elif type_str == "hemi":
            type_val = LightType.HEMI
        else:
            raise LoadError(f'Unrecognized light type "{type_str}"')
        shadow = bool(light.get("shadow", False))
        direction = _normalize(read_vector3(light.get("direction")))
        intensity = require_number(light, "strength")
        out.append(
            Light(type=type_val, shadow=shadow, direction=direction, intensity=intensity)
        )
    return out
