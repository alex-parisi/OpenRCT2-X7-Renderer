"""
Shared command-line scaffolding for object generators.
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import LoadError, parse_config
from .constants import TILE_SIZE
from .lights import default_lights, load_lights
from .ray_trace import Context
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


def make_context(lights: list[Light], units_per_tile: float, test: bool) -> Context:
    """Build a render Context whose camera scale is driven by the object's
    configured `units_per_tile`. Test mode scales ``upt`` down by ``TEST_ZOOM``
    (0.125×), zooming in to show material detail in a single-viewpoint preview."""
    upt = TEST_ZOOM * units_per_tile if test else units_per_tile
    return Context(lights=lights, dither=True, upt=upt)


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
