"""Tests for the shared config parsing + validation helpers."""

import numpy as np
import pytest
from openrct2_x7_renderer.config import (
    LoadError,
    as_array_or_wrap,
    load_meshes,
    load_preview,
    optional_bool,
    optional_int,
    optional_number,
    optional_string,
    optional_string_list,
    parse_config,
    read_vector3,
    require_int,
    require_number,
    require_string,
)


def test_parse_config_json(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('{"name": "x", "n": 3}')
    assert parse_config(p) == {"name": "x", "n": 3}


def test_parse_config_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: x\nn: 3\n")
    assert parse_config(p) == {"name": "x", "n": 3}


def test_parse_config_extension_picks_format(tmp_path):
    # A .yml file with flow-style content still parses (YAML is a JSON superset).
    p = tmp_path / "c.yml"
    p.write_text("{a: 1, b: [2, 3]}")
    assert parse_config(p) == {"a": 1, "b": [2, 3]}


def test_parse_config_rejects_non_object_root(tmp_path):
    p = tmp_path / "c.json"
    p.write_text("[1, 2, 3]")
    with pytest.raises(LoadError, match="not an object"):
        parse_config(p)


def test_parse_config_yaml_without_pyyaml_raises_load_error(tmp_path, monkeypatch):
    import sys

    p = tmp_path / "c.yaml"
    p.write_text("a: 1\n")
    # Poison the yaml entry in sys.modules so `import yaml` raises ImportError.
    monkeypatch.setitem(sys.modules, "yaml", None)
    with pytest.raises(LoadError, match="PyYAML is required"):
        parse_config(p)


def test_parse_config_accepts_str_path(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('{"ok": true}')
    assert parse_config(str(p)) == {"ok": True}


def test_require_string_ok_and_missing():
    assert require_string({"k": "v"}, "k") == "v"
    with pytest.raises(LoadError):
        require_string({}, "k")
    with pytest.raises(LoadError):
        require_string({"k": 3}, "k")


def test_optional_string_default_and_type_check():
    assert optional_string({}, "k") == ""
    assert optional_string({}, "k", "fallback") == "fallback"
    assert optional_string({"k": "v"}, "k") == "v"
    with pytest.raises(LoadError):
        optional_string({"k": 1}, "k")


def test_optional_string_list_forms():
    assert optional_string_list({}, "k") == []
    assert optional_string_list({"k": "one"}, "k") == ["one"]
    assert optional_string_list({"k": ["a", "b"]}, "k") == ["a", "b"]


def test_optional_string_list_rejects_mixed():
    with pytest.raises(LoadError):
        optional_string_list({"k": ["a", 2]}, "k")
    with pytest.raises(LoadError):
        optional_string_list({"k": 5}, "k")


def test_require_int_rejects_bool():
    # bool is an int subclass in Python; the loader must reject it explicitly.
    assert require_int({"k": 4}, "k") == 4
    with pytest.raises(LoadError):
        require_int({"k": True}, "k")
    with pytest.raises(LoadError):
        require_int({"k": 1.5}, "k")
    with pytest.raises(LoadError):
        require_int({}, "k")


def test_optional_int_default_and_bool_rejection():
    assert optional_int({}, "k", 7) == 7
    assert optional_int({"k": 2}, "k", 7) == 2
    with pytest.raises(LoadError):
        optional_int({"k": False}, "k", 7)


def test_require_number_accepts_int_and_float_not_bool():
    assert require_number({"k": 2}, "k") == 2.0
    assert require_number({"k": 2.5}, "k") == 2.5
    assert isinstance(require_number({"k": 2}, "k"), float)
    with pytest.raises(LoadError):
        require_number({"k": True}, "k")
    with pytest.raises(LoadError):
        require_number({}, "k")


def test_optional_number_default():
    assert optional_number({}, "k", 1.25) == 1.25
    assert optional_number({"k": 3}, "k", 1.25) == 3.0
    with pytest.raises(LoadError):
        optional_number({"k": "x"}, "k", 1.25)


def test_optional_bool():
    assert optional_bool({}, "k") is False
    assert optional_bool({}, "k", True) is True
    assert optional_bool({"k": True}, "k") is True
    with pytest.raises(LoadError):
        optional_bool({"k": 1}, "k")


def test_read_vector3_ok():
    v = read_vector3([1, 2, 3])
    assert isinstance(v, np.ndarray)
    assert v.dtype == np.float64
    assert np.array_equal(v, [1.0, 2.0, 3.0])


def test_read_vector3_rejects_wrong_length():
    with pytest.raises(LoadError):
        read_vector3([1, 2])
    with pytest.raises(LoadError):
        read_vector3("not a list")


def test_read_vector3_rejects_non_numeric_element():
    with pytest.raises(LoadError, match="not a number"):
        read_vector3([1, "bad", 3])


def test_as_array_or_wrap():
    assert as_array_or_wrap([1, 2]) == [1, 2]
    assert as_array_or_wrap("scalar") == ["scalar"]
    assert as_array_or_wrap({"a": 1}) == [{"a": 1}]
    with pytest.raises(LoadError):
        as_array_or_wrap(None)
    with pytest.raises(LoadError):
        as_array_or_wrap([])


def test_load_meshes_returns_list(tmp_path):
    # Write a minimal OBJ so load_mesh succeeds.
    obj = tmp_path / "m.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    meshes = load_meshes({"meshes": [str(obj)]})
    assert len(meshes) == 1


def test_load_meshes_rejects_non_array():
    with pytest.raises(LoadError, match='"meshes"'):
        load_meshes({"meshes": "notanarray"})


def test_load_meshes_rejects_non_string_path():
    with pytest.raises(LoadError, match="not a string"):
        load_meshes({"meshes": [42]})


def test_load_meshes_missing_key():
    with pytest.raises(LoadError):
        load_meshes({})


def test_load_preview_returns_none_when_absent():
    assert load_preview({}) is None


def test_load_preview_rejects_non_string():
    with pytest.raises(LoadError, match='"preview"'):
        load_preview({"preview": 123})


def test_load_preview_raises_on_bad_file():
    with pytest.raises(LoadError, match="Unable to open"):
        load_preview({"preview": "/nonexistent/path/image.png"})


def test_load_preview_success(tmp_path):
    from openrct2_x7_renderer.image import write_png
    from openrct2_x7_renderer.types import IndexedImage

    img = IndexedImage.blank(4, 4)
    png_path = tmp_path / "preview.png"
    write_png(img, png_path)
    result = load_preview({"preview": str(png_path)})
    assert result is not None
    assert result.width == 4
