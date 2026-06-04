"""
Rendering constants shared by the vehicle and scenery generators.
Ported from X7's rendering engine
https://github.com/X123M3-256/RCTGen
"""

TILE_SIZE = 3.3

RENDER_WIDTH = 255
RENDER_HEIGHT = 256
UNITS_PER_TILE = 4096
UNITS_PER_PIXEL = 128
FRAGMENT_UNUSED = 255
REGION_MASK = 0x7
MAX_REGIONS = 8


# Material flags (from src/iso-render/Mesh.hpp).
MATERIAL_HAS_TEXTURE = 1 << 0
MATERIAL_IS_REMAPPABLE = 1 << 1
MATERIAL_IS_MASK = 1 << 2
MATERIAL_NO_AO = 1 << 3
MATERIAL_BACKGROUND_AA = 1 << 4
MATERIAL_BACKGROUND_AA_DARK = 1 << 5
MATERIAL_IS_VISIBLE_MASK = 1 << 6
MATERIAL_NO_BLEED = 1 << 7
MATERIAL_IS_FLAT_SHADED = 1 << 8


# Mesh flags (RayTrace.hpp).
MESH_MASK = 1 << 0
MESH_GHOST = 1 << 1


# Light types (Renderer.hpp).
LIGHT_HEMI = 0
LIGHT_DIFFUSE = 1
LIGHT_SPECULAR = 2


# AA / AO sample counts (Renderer.cpp).
AA_NUM_SAMPLES_U = 4
AA_NUM_SAMPLES_V = 4
AO_NUM_SAMPLES_U = 8
AO_NUM_SAMPLES_V = 4
