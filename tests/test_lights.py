"""Tests for the default light rig and the lights config loader."""

import logging

import numpy as np
import pytest
from openrct2_x7_renderer.config import LoadError
from openrct2_x7_renderer.constants import LightType
from openrct2_x7_renderer.lights import default_lights, load_lights


def test_default_lights_count_and_types():
    lights = default_lights()
    assert len(lights) == 9
    # Exactly one specular light in the default rig (index 2).
    assert sum(1 for light in lights if light.type == LightType.SPECULAR) == 1
    assert lights[2].type == LightType.SPECULAR


def test_default_lights_directions_are_unit_length():
    for light in default_lights():
        assert np.isclose(np.linalg.norm(light.direction), 1.0)


def test_load_lights_diffuse_and_specular():
    cfg = [
        {"type": "diffuse", "direction": [0, 1, 0], "strength": 0.5},
        {"type": "specular", "direction": [0, 0, 2], "strength": 0.9, "shadow": True},
    ]
    lights = load_lights(cfg)
    assert len(lights) == 2
    assert lights[0].type == LightType.DIFFUSE
    assert lights[0].shadow == 0
    assert lights[0].intensity == 0.5
    assert lights[1].type == LightType.SPECULAR
    assert lights[1].shadow == 1


def test_load_lights_normalizes_direction():
    lights = load_lights([{"type": "diffuse", "direction": [0, 0, 5], "strength": 1.0}])
    assert np.allclose(lights[0].direction, [0.0, 0.0, 1.0])


def test_load_lights_rejects_non_array():
    with pytest.raises(LoadError, match="not an array"):
        load_lights({"type": "diffuse"})


def test_load_lights_rejects_unknown_type():
    with pytest.raises(LoadError, match="Unrecognized light type"):
        load_lights([{"type": "ambient", "direction": [0, 1, 0], "strength": 1.0}])


def test_load_lights_requires_strength():
    with pytest.raises(LoadError):
        load_lights([{"type": "diffuse", "direction": [0, 1, 0]}])


def test_load_lights_hemi_type():
    lights = load_lights([{"type": "hemi", "direction": [0, 1, 0], "strength": 0.3}])
    assert len(lights) == 1
    assert lights[0].type == LightType.HEMI


def test_load_lights_skips_non_object_elements(caplog):
    with caplog.at_level(logging.WARNING, logger="openrct2_x7_renderer.lights"):
        lights = load_lights(
            [
                "garbage",
                {"type": "diffuse", "direction": [0, 1, 0], "strength": 1.0},
            ]
        )
    assert len(lights) == 1
    assert "non-object element" in caplog.text


def test_load_lights_empty_list():
    assert load_lights([]) == []


def test_normalize_rejects_zero_vector():
    """_normalize raises ValueError when the direction is the zero vector."""
    with pytest.raises(ValueError, match="zero vector"):
        load_lights([{"type": "diffuse", "direction": [0, 0, 0], "strength": 1.0}])
