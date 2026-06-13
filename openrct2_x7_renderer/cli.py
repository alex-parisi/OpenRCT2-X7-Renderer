"""
Shared command-line scaffolding for object generators.
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import LoadError, parse_config
from .lights import default_lights, load_lights
from .ray_trace import DITHER_MODES, Context
from .remap import load_remap_overrides
from .types import Light

TEST_ZOOM = 0.125

RenderFn = Callable[[argparse.Namespace, dict[str, Any], list[Light]], None]


def parse_cli_args(prog: str, argv: list[str] | None) -> argparse.Namespace:
    """Parse the flags shared by both generators: `--test` / `--skip-render`
    (mutually exclusive) and a single config-file `input` path."""
    parser = argparse.ArgumentParser(prog=prog)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--test", action="store_true", help="single-viewpoint render to test/")
    group.add_argument(
        "--skip-render", action="store_true", help="reuse previously rendered sprites"
    )
    parser.add_argument("input", type=Path)
    return parser.parse_args(argv)


def output_directory_of(root: dict[str, Any]) -> Path:
    """The config's `output_directory`, or the current directory if unset."""
    out = root.get("output_directory")
    return Path(out) if isinstance(out, str) else Path(".")


def make_context(
    lights: list[Light],
    units_per_tile: float,
    test: bool,
    root: dict[str, Any] | None = None,
    dither: bool | str | None = None,
    stability: float | None = None,
) -> Context:
    """Build a render Context whose camera scale is driven by the object's
    configured `units_per_tile`. Test mode scales ``upt`` down by ``TEST_ZOOM``
    (0.125×), zooming in to show material detail in a single-viewpoint preview.

    In test mode, when ``root`` (the parsed config) is supplied, an optional
    ``test_remap_colors`` block is read and applied to ``render_view`` output so
    the preview shows repaint colours instead of the raw remap windows. Outside
    test mode the overrides are ignored, so real renders keep their remap
    windows intact for OpenRCT2 to repaint.

    Dithering is on by default. An explicit ``dither`` argument (used by the
    Blender add-ons to pass a UI selection) takes precedence; otherwise an
    optional top-level ``dither`` knob in the config is read. Both accept a
    boolean (matching the original RCTGen tools, where ``true`` means
    Floyd-Steinberg) or a mode string: ``"none"``, ``"floyd_steinberg"``,
    ``"bayer"`` (a frame-stable ordered dither suited to animated objects), or
    ``"blue_noise"`` (a frame-stable blue-noise dither with less visible residual
    motion under rotation).

    A ``dither_stability`` knob (a non-negative number, default ``0``) sets a
    temporal-stability deadband in 8-bit colour units: shading changes smaller
    than it quantise identically between frames, further suppressing "swimming".
    An explicit ``stability`` argument takes precedence over the config knob."""
    upt = TEST_ZOOM * units_per_tile if test else units_per_tile
    overrides = load_remap_overrides(root) if (test and root is not None) else {}
    if dither is None:
        dither = True
        if root is not None and "dither" in root:
            value = root["dither"]
            if isinstance(value, bool):
                dither = value
            elif isinstance(value, str) and value in DITHER_MODES:
                dither = value
            else:
                raise LoadError(
                    'Property "dither" must be a boolean or one of: '
                    + ", ".join(sorted(DITHER_MODES))
                )
    if stability is None:
        stability = 0.0
        if root is not None and "dither_stability" in root:
            value = root["dither_stability"]
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                stability = float(value)
            else:
                raise LoadError('Property "dither_stability" must be a non-negative number')
    return Context(
        lights=lights, dither=dither, upt=upt, remap_overrides=overrides, stability=stability
    )


def run_cli(prog: str, argv: list[str] | None, render: RenderFn) -> int:
    """Run the shared CLI flow and return a process exit code.

    Parses the args, reads the config, resolves the lights (the config's
    `lights` block if present, else the default rig), then hands off to
    `render`. Any failure is reported on stderr and yields exit code 1.
    """
    args = parse_cli_args(prog, argv)

    try:
        root = parse_config(args.input)
        lights = load_lights(root["lights"]) if "lights" in root else default_lights()
        render(args, root, lights)
    except (LoadError, OSError, ValueError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0
