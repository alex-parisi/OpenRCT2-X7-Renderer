"""Tests for the preview-only remap recolouring (openrct2_x7_renderer.remap)."""

import numpy as np
import pytest
from openrct2_x7_renderer.remap import (
    REMAP_COLOR_RAMPS,
    REMAP_WINDOWS,
    apply_remap_overrides,
    load_remap_overrides,
)
from openrct2_x7_renderer.types import LoadError

# ---------- table sanity ----------


def test_every_ramp_has_twelve_shades_in_palette_range():
    for name, ramp in REMAP_COLOR_RAMPS.items():
        assert len(ramp) == 12, name
        assert all(0 <= idx <= 255 for idx in ramp), name


# ---------- load_remap_overrides ----------


def test_load_returns_empty_when_block_absent():
    assert load_remap_overrides({}) == {}


def test_load_resolves_colour_names_to_ramps():
    out = load_remap_overrides({"test_remap_colors": {"1": "bordeaux_red"}})
    assert out == {1: REMAP_COLOR_RAMPS["bordeaux_red"]}


def test_load_accepts_integer_region_keys():
    # YAML parses bare `1:` as an int key; JSON gives "1". Both must work.
    out = load_remap_overrides({"test_remap_colors": {1: "dark_green", 3: "yellow"}})
    assert set(out) == {1, 3}


def test_load_rejects_non_object_block():
    with pytest.raises(LoadError):
        load_remap_overrides({"test_remap_colors": ["bordeaux_red"]})


def test_load_rejects_unknown_region():
    with pytest.raises(LoadError):
        load_remap_overrides({"test_remap_colors": {"4": "grey"}})


def test_load_rejects_non_numeric_region():
    with pytest.raises(LoadError):
        load_remap_overrides({"test_remap_colors": {"primary": "grey"}})


def test_load_rejects_unknown_colour():
    with pytest.raises(LoadError):
        load_remap_overrides({"test_remap_colors": {"1": "chartreuse"}})


def test_load_rejects_non_string_colour():
    with pytest.raises(LoadError):
        load_remap_overrides({"test_remap_colors": {"1": 42}})


# ---------- apply_remap_overrides ----------


def test_apply_is_noop_without_overrides():
    pixels = np.array([[0, 46, 243]], dtype=np.uint8)
    out = apply_remap_overrides(pixels, {})
    assert out is pixels  # same object, no copy taken


def test_apply_maps_each_shade_to_target_ramp():
    start = REMAP_WINDOWS[1]
    ramp = REMAP_COLOR_RAMPS["grey"]
    window = np.array([[start + k for k in range(12)]], dtype=np.uint8)
    out = apply_remap_overrides(window, {1: ramp})
    assert out.tolist() == [list(ramp)]


def test_apply_leaves_non_window_pixels_untouched():
    # Transparent (0) and a non-remap diffuse index must be preserved.
    pixels = np.array([[0, 100, REMAP_WINDOWS[2]]], dtype=np.uint8)
    out = apply_remap_overrides(pixels, {2: REMAP_COLOR_RAMPS["teal"]})
    assert out[0, 0] == 0
    assert out[0, 1] == 100
    assert out[0, 2] == REMAP_COLOR_RAMPS["teal"][0]


def test_apply_does_not_mutate_input():
    pixels = np.array([[REMAP_WINDOWS[3]]], dtype=np.uint8)
    apply_remap_overrides(pixels, {3: REMAP_COLOR_RAMPS["yellow"]})
    assert pixels[0, 0] == REMAP_WINDOWS[3]


def test_apply_reads_from_original_so_overlapping_targets_dont_double_remap():
    # dark_pink's ramp lands in region 2's own window (202+). Region 3 pixels
    # remapped onto that range must not then be re-remapped by region 2.
    region3_window = np.array(
        [[REMAP_WINDOWS[3] + k for k in range(12)]], dtype=np.uint8
    )
    overrides = {
        2: REMAP_COLOR_RAMPS["grey"],
        3: REMAP_COLOR_RAMPS["dark_pink"],  # outputs into 202..208
    }
    out = apply_remap_overrides(region3_window, overrides)
    # Each region-3 pixel maps straight to dark_pink's ramp, untouched by region 2.
    assert out.tolist() == [list(REMAP_COLOR_RAMPS["dark_pink"])]
