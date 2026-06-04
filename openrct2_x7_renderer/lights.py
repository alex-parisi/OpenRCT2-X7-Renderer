"""
Default light rig + light-config loading, shared by the generators.
Ported from X7's rendering engine
https://github.com/X123M3-256/RCTGen
"""

from typing import Any

import numpy as np

from .config import LoadError, read_vector3, require_number, require_string
from .constants import LIGHT_DIFFUSE, LIGHT_SPECULAR
from .types import Light


def _normalize(v: list[float]) -> np.ndarray:
    arr = np.array(v, dtype=np.float64)
    n = np.linalg.norm(arr)
    if n > 0:
        arr = arr / n
    return arr


def default_lights() -> list[Light]:
    return [
        Light(LIGHT_DIFFUSE, 0, _normalize([0.0, -1.0, 0.0]), 0.1),
        Light(LIGHT_DIFFUSE, 0, _normalize([0.0, 0.5, -1.0]), 0.8),
        Light(LIGHT_SPECULAR, 1, _normalize([1.0, 1.65, -1.0]), 0.5),
        Light(LIGHT_DIFFUSE, 1, _normalize([1.0, 1.7, -1.0]), 0.8),
        Light(LIGHT_DIFFUSE, 0, np.array([0.0, 1.0, 0.0], dtype=np.float64), 0.45),
        Light(LIGHT_DIFFUSE, 0, _normalize([-1.0, 0.85, 1.0]), 0.475),
        Light(LIGHT_DIFFUSE, 0, _normalize([0.75, 0.4, -1.0]), 0.6),
        Light(LIGHT_DIFFUSE, 0, _normalize([1.0, 0.25, 0.0]), 0.5),
        Light(LIGHT_DIFFUSE, 0, _normalize([-1.0, -0.5, 0.0]), 0.1),
    ]


def load_lights(value: Any) -> list[Light]:
    if not isinstance(value, list):
        raise LoadError('"lights" is not an array')
    out: list[Light] = []
    for light in value:
        if not isinstance(light, dict):
            print("Warning: Light array contains a non-object element — ignoring")
            continue
        type_str = require_string(light, "type")
        if type_str == "diffuse":
            type_val = LIGHT_DIFFUSE
        elif type_str == "specular":
            type_val = LIGHT_SPECULAR
        else:
            raise LoadError(f'Unrecognized light type "{type_str}"')
        shadow = light.get("shadow", False)
        direction = read_vector3(light.get("direction"))
        n = np.linalg.norm(direction)
        if n > 0:
            direction = direction / n
        intensity = require_number(light, "strength")
        out.append(
            Light(type=type_val, shadow=int(shadow), direction=direction, intensity=intensity)
        )
    return out
