"""Tests for the shared CLI scaffolding (parse_cli_args, run_cli, etc.)."""

from pathlib import Path

import pytest
from openrct2_x7_renderer.cli import (
    TEST_ZOOM,
    make_context,
    output_directory_of,
    parse_cli_args,
    run_cli,
)
from openrct2_x7_renderer.config import LoadError
from openrct2_x7_renderer.lights import default_lights

# ---------- parse_cli_args ----------


def test_parse_cli_args_input_only(tmp_path):
    p = tmp_path / "cfg.json"
    p.touch()
    args = parse_cli_args("test-prog", [str(p)])
    assert args.input == p
    assert not args.test
    assert not args.skip_render


def test_parse_cli_args_test_flag(tmp_path):
    p = tmp_path / "cfg.json"
    p.touch()
    args = parse_cli_args("test-prog", ["--test", str(p)])
    assert args.test
    assert not args.skip_render


def test_parse_cli_args_skip_render_flag(tmp_path):
    p = tmp_path / "cfg.json"
    p.touch()
    args = parse_cli_args("test-prog", ["--skip-render", str(p)])
    assert args.skip_render
    assert not args.test


def test_parse_cli_args_test_and_skip_render_are_mutually_exclusive(tmp_path):
    p = tmp_path / "cfg.json"
    p.touch()
    with pytest.raises(SystemExit):
        parse_cli_args("test-prog", ["--test", "--skip-render", str(p)])


# ---------- output_directory_of ----------


def test_output_directory_of_uses_config_value():
    root = {"output_directory": "/some/path"}
    assert output_directory_of(root) == Path("/some/path")


def test_output_directory_of_defaults_to_cwd():
    assert output_directory_of({}) == Path(".")
    assert output_directory_of({"output_directory": 42}) == Path(".")


# ---------- make_context ----------


def test_make_context_uses_upt_directly_when_not_test():
    lights = default_lights()
    ctx = make_context(lights, 16.0, test=False)
    assert ctx.upt == 16.0


def test_make_context_scales_upt_in_test_mode():
    lights = default_lights()
    ctx = make_context(lights, 16.0, test=True)
    assert ctx.upt == pytest.approx(TEST_ZOOM * 16.0)


def test_make_context_loads_remap_overrides_in_test_mode():
    from openrct2_x7_renderer.remap import REMAP_COLOR_RAMPS

    root = {"test_remap_colors": {"1": "bordeaux_red"}}
    ctx = make_context(default_lights(), 16.0, test=True, root=root)
    assert ctx.remap_overrides == {1: REMAP_COLOR_RAMPS["bordeaux_red"]}


def test_make_context_ignores_remap_overrides_outside_test_mode():
    # Real renders must keep their raw remap windows for OpenRCT2 to repaint.
    root = {"test_remap_colors": {"1": "bordeaux_red"}}
    ctx = make_context(default_lights(), 16.0, test=False, root=root)
    assert ctx.remap_overrides == {}


def test_make_context_without_root_has_no_overrides():
    ctx = make_context(default_lights(), 16.0, test=True)
    assert ctx.remap_overrides == {}


def test_make_context_dithers_by_default():
    assert (
        make_context(default_lights(), 16.0, test=False).dither_mode
        == "floyd_steinberg"
    )
    assert (
        make_context(default_lights(), 16.0, test=False, root={}).dither_mode
        == "floyd_steinberg"
    )


def test_make_context_dither_can_be_disabled_by_config():
    ctx = make_context(default_lights(), 16.0, test=False, root={"dither": False})
    assert ctx.dither_mode == "none"


def test_make_context_dither_mode_from_config_string():
    ctx = make_context(default_lights(), 16.0, test=False, root={"dither": "bayer"})
    assert ctx.dither_mode == "bayer"


def test_make_context_dither_rejects_invalid_value():
    with pytest.raises(LoadError, match="dither"):
        make_context(default_lights(), 16.0, test=False, root={"dither": "yes"})


def test_make_context_explicit_dither_arg_overrides_config():
    # The add-ons pass the UI selection explicitly; it wins over the config and
    # is used verbatim without config validation.
    ctx = make_context(
        default_lights(), 16.0, test=False, root={"dither": False}, dither="bayer"
    )
    assert ctx.dither_mode == "bayer"


# ---------- run_cli ----------


def test_run_cli_success(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"name": "test"}')

    called = []

    def render(args, root, lights):
        called.append((args, root, lights))

    code = run_cli("prog", [str(cfg)], render)
    assert code == 0
    assert len(called) == 1
    assert called[0][1] == {"name": "test"}


def test_run_cli_uses_default_lights_when_no_lights_key(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}")

    captured = []

    def render(args, root, lights):
        captured.append(lights)

    run_cli("prog", [str(cfg)], render)
    assert len(captured[0]) == 9  # default rig has 9 lights


def test_run_cli_uses_config_lights(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"lights": [{"type": "diffuse", "direction": [0, 1, 0], "strength": 0.5}]}')

    captured = []

    def render(args, root, lights):
        captured.append(lights)

    run_cli("prog", [str(cfg)], render)
    assert len(captured[0]) == 1


def test_run_cli_returns_1_on_render_error(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}")

    def render(args, root, lights):
        raise RuntimeError("boom")

    code = run_cli("prog", [str(cfg)], render)
    assert code == 1


def test_run_cli_returns_1_on_missing_config(tmp_path):
    code = run_cli("prog", [str(tmp_path / "missing.json")], lambda *a: None)
    assert code == 1


def test_run_cli_prints_error_on_stderr(tmp_path, capsys):
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}")

    run_cli("prog", [str(cfg)], lambda *a: (_ for _ in ()).throw(ValueError("oops")))
    err = capsys.readouterr().err
    assert "Error:" in err
    assert "oops" in err
