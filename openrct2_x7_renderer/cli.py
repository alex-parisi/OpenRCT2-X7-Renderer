"""
Shared command-line scaffolding for the vehicle and scenery generators.

Both front-ends parse the same flags, read a config file, resolve the light
rig, and build a render Context whose camera scale comes from the loaded
object. `run_cli` wraps that common flow (and its error handling) so each
entry point only has to say how to load and export its own object type.
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from .config import parse_config
from .lights import default_lights, load_lights
from .ray_trace import Context
from .types import Light

# Test mode renders at 8x zoom for fast eyeballing of a single viewpoint.
TEST_ZOOM = 0.125

# Given the parsed CLI args, the raw config dict, and the resolved lights,
# load and export the object. Raising LoadError (or any exception) is turned
# into a non-zero exit code with an "Error: ..." message.
RenderFn = Callable[[argparse.Namespace, dict, list[Light]], None]


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


def output_directory_of(root: dict) -> Path:
    """The config's `output_directory`, or the current directory if unset."""
    out = root.get("output_directory")
    return Path(out) if isinstance(out, str) else Path(".")


def make_context(lights: list[Light], units_per_tile: float, test: bool) -> Context:
    """Build a render Context whose camera scale is driven by the object's
    configured `units_per_tile`. Test mode zooms in for fast iteration."""
    upt = TEST_ZOOM * units_per_tile if test else units_per_tile
    return Context.make(lights=lights, dither=True, upt=upt)


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
    except Exception as e:  # LoadError is the expected case; report any failure cleanly
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0
