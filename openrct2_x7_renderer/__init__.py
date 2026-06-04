"""
OpenRCT2 X7 renderer: the Embree-backed ray tracer (``_x7_renderer`` C
extension) plus the OBJ/mesh, palette, image, ``images.dat``, lighting, and
geometry primitives that turn meshes into OpenRCT2 palette-indexed sprites.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("openrct2-x7-renderer")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"
