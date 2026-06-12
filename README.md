# OpenRCT2 X7 Renderer

`openrct2-x7-renderer` is the Embree-backed isometric renderer that turns
triangle meshes into OpenRCT2 palette-indexed sprites.

Heavily inspired by X7's [RCTGen](https://github.com/X123M3-256/RCTGen).

## Install

Requires **Python 3.11+**. Embree-vendored wheels are published to PyPI:

```bash
pip install openrct2-x7-renderer
```

The wheels bundle Embree + TBB, so there is no system Embree dependency at
runtime.

## Build from source

Building the extension needs [Embree 4](https://github.com/RenderKit/embree)
and a C++23 compiler. On macOS: `brew install embree`.

```bash
uv sync
uv run pytest   # coverage is enabled by default via pyproject.toml
```

## Quick start

```python
import numpy as np

from openrct2_x7_renderer.mesh import load_mesh
from openrct2_x7_renderer.lights import default_lights
from openrct2_x7_renderer.ray_trace import Context, VIEWS

mesh = load_mesh("model.obj")
ctx = Context(lights=default_lights(), dither=True)
with ctx.begin_render() as scene:
    with scene.add_model(mesh, matrix=np.eye(3), translation=np.zeros(3)).finalize() as ready:
        sprite = ready.render_view(VIEWS[0])   # -> IndexedImage
```

## Usage

### Render context

`Context` is the entry point for rendering. The three constructor parameters
control the entire render:

```python
from openrct2_x7_renderer.ray_trace import Context
from openrct2_x7_renderer.lights import default_lights
from openrct2_x7_renderer.constants import TILE_SIZE

ctx = Context(
    lights=default_lights(),  # light rig (see below)
    dither=True,              # Floyd-Steinberg dithering when quantizing to the palette
    upt=TILE_SIZE,            # camera scale: units-per-tile (default 3.3)
)
```

`upt` controls zoom: smaller values zoom in, larger values zoom out.  The CLI
helpers scale `upt` by `TEST_ZOOM` (0.125) in test mode (`--test`) for fast
iteration.

### Render lifecycle (typestate pattern)

A single render pass follows a strict state machine enforced at the type level:

1. `Context.begin_render()` → returns a `SceneBuilder`
2. `SceneBuilder.add_model(...)` → chainable, returns `self`
3. `SceneBuilder.finalize()` → returns a `FinalizedScene`
4. `FinalizedScene.render_view(view)` / `.render_silhouette(view)` → produces sprites
5. Cleanup via `FinalizedScene` context manager or explicit `end_render()` call

```python
ctx = Context(lights=default_lights())

with ctx.begin_render() as scene:
    with (
        scene
        .add_model(mesh, matrix=np.eye(3), translation=np.zeros(3))
        .add_model(mesh2, matrix=rotation, translation=offset, mask=MeshFlag.GHOST)
        .finalize()
    ) as ready:
        sprite = ready.render_view(VIEWS[0])           # full shaded render
        mask   = ready.render_silhouette(VIEWS[0])     # solid silhouette / mask sprite
```

`FinalizedScene.__exit__` calls `end_render()`, freeing the Embree scene.
`SceneBuilder.__exit__` calls `end_render()` only when `finalize()` was **not**
called (e.g. the block exited early due to an exception). The context may be
reused across multiple render passes by calling `begin_render()` again.

`SceneBuilder.add_model` parameters:

| Parameter | Type | Description |
|---|---|---|
| `mesh` | `Mesh` | Triangle mesh returned by `load_mesh` |
| `matrix` | `(3,3) float64` | Rotation / orientation matrix applied to the mesh |
| `translation` | `(3,) float64` | World-space offset after rotation |
| `mask` | `int` | Bitmask of `MeshFlag` values (default `0`) |

`MeshFlag.GHOST` (from `constants`) makes the mesh transparent (ghost ride
vehicles). `MeshFlag.MASK` marks the mesh as a collision/mask geometry.

### Views

`VIEWS` is a tuple of four `(3,3)` world-rotation matrices — one per compass
corner, matching OpenRCT2's four viewpoints (NE, NW, SW, SE). These rotate
the scene before the dimetric projection:

```python
from openrct2_x7_renderer.ray_trace import Context, VIEWS

ctx = Context(lights=default_lights())
with ctx.begin_render() as scene:
    with scene.add_model(mesh, np.eye(3), np.zeros(3)).finalize() as ready:
        for view in VIEWS:
            sprite = ready.render_view(view)
```

Pass any custom `(3,3) float64` orthonormal matrix to render from an arbitrary
direction.

### Lights

`default_lights()` returns a nine-light rig that matches X7's RCTGen defaults.
To customise the rig, build a list of `Light` objects:

```python
from openrct2_x7_renderer.types import Light
from openrct2_x7_renderer.constants import LightType
import numpy as np

lights = [
    Light(
        type=LightType.DIFFUSE,                      # LightType.DIFFUSE | .SPECULAR | .HEMI
        shadow=True,                                 # whether the light casts shadows
        direction=np.array([1.0, 1.65, -1.0]) / ..., # unit vector toward the light
        intensity=0.8,                               # strength multiplier
    ),
]
```

Light types:

| Constant | Behaviour |
|---|---|
| `LightType.DIFFUSE` | Lambertian diffuse shading |
| `LightType.SPECULAR` | Phong specular highlight |
| `LightType.HEMI` | Hemisphere / sky light |

When loading lights from a config via `load_lights`, each `direction` must be
a non-zero vector; passing `[0, 0, 0]` raises `ValueError`.

Lights can also be loaded from a config file (see [Config files](#config-files)).

### Mesh loading

```python
from openrct2_x7_renderer.mesh import load_mesh
import numpy as np

mesh = load_mesh("model.obj")

# Optional: apply an orthonormal transform at load time
# (e.g. mirror or axis-swap).  Negative determinant flips winding order.
# Raises LoadError if the matrix is not orthonormal (|det| − 1 > 0.001).
mesh = load_mesh("model.obj", transform=np.diag([-1, 1, 1]))
```

`load_mesh` parses the OBJ file and the MTL it references, loading textures
(`map_Kd`) as linear-RGB float32 arrays. Standard MTL properties are honoured.

**Normal handling:** if the OBJ defines `vn` entries _and_ every face vertex
references one, the artist normals are used directly. If normals are defined
but any face vertex omits the `//vn` token, a `WARNING` is logged and
area-weighted face normals are generated for the entire mesh. If no `vn`
entries exist at all, face normals are generated silently.

**UV handling:** if a face vertex references a texture coordinate index but no
`vt` entries are defined, a `WARNING` is logged and that vertex's UV defaults
to `(0, 0)`.

| MTL directive | Effect |
|---|---|
| `Kd r g b` | Diffuse colour (sRGB → linear on load; used when no `map_Kd` is present) |
| `Ks r g b` | Specular reflectance (sRGB → linear on load) |
| `Ka r g b` | Ambient reflectance (sRGB → linear on load) |
| `Ns value` | Phong shininess exponent |
| `map_Kd file` | Diffuse texture (sRGB → linear on load) |

### Material name flags

The renderer recognises keywords in OBJ material names to set rendering
behaviour. Keywords are matched by substring, case-sensitive:

| Keyword in material name | Effect |
|---|---|
| `Remap1`, `Remap2`, `Remap3` | Map diffuse colour into OpenRCT2 remap palette region 1–3 (runtime recoloring) |
| `Greyscale` | Use palette region 4 (greyscale remap) |
| `Peep` | Use palette region 5 (peep skin remap) |
| `Glass` | Render in the translucent glass pass |
| `Ghost` | Ghost geometry: primary rays trace through it (not drawn) but it still contributes to the silhouette/occlusion pass |
| `Back` | Included only in rear-wall sprite blocks |
| `Front` | Included only in front-wall sprite blocks |
| `Mask` | Treated as a visibility mask (`MaterialFlag.IS_MASK`) |
| `NoAO` | Disable ambient occlusion for this material |
| `Edge` | Enable background anti-aliasing blending |
| `DarkEdge` | Enable dark-variant background AA blending |
| `NoBleed` | Disable colour bleed from neighbouring pixels |

Example MTL material name: `mat_Remap1_NoAO` gets remappable region 1 with AO
disabled.

### Constants

All material/mesh/light constants are exposed as strongly-typed enums:

```python
from openrct2_x7_renderer.constants import MaterialFlag, MeshFlag, LightType

# MaterialFlag is an IntFlag — supports bitwise operations
flags = MaterialFlag.IS_REMAPPABLE | MaterialFlag.NO_AO

# MeshFlag is an IntFlag
mask = MeshFlag.GHOST | MeshFlag.MASK

# LightType is an IntEnum
light_type = LightType.DIFFUSE
```

### Silhouette rendering

`FinalizedScene.render_silhouette` produces a solid silhouette sprite — every
hit pixel is rendered as flat mid-gray (linear 0.5, 0.5, 0.5) and quantized to
the nearest RCT2 palette entry; transparent pixels are unchanged:

```python
with ctx.begin_render() as scene:
    with scene.add_model(mesh, np.eye(3), np.zeros(3)).finalize() as ready:
        mask_sprite = ready.render_silhouette(VIEWS[0])
```

### Image I/O

```python
from openrct2_x7_renderer.image import write_png, read_png, quantize_to_indexed, PREVIEW_SIZE

# Write a rendered IndexedImage as a paletted PNG (transparent index = 0).
write_png(sprite, "out.png")

# Read a paletted PNG back (must already be an 8-bit palette-mode PNG).
img = read_png("existing.png")

# Convert any image format to an IndexedImage sized for a 112x112 preview.
preview = quantize_to_indexed("photo.jpg", size=PREVIEW_SIZE)
```

`quantize_to_indexed` resizes with Lanczos resampling, Floyd-Steinberg dithers
into the safe palette ranges (indices 10–201, 214–226, and 240–242 — excluding
the remap windows and animated colours), and maps alpha < 128 to transparent.

### Geometry helpers

```python
from openrct2_x7_renderer.geometry import (
    rotate_x, rotate_y, rotate_z,
    combine_model_world,
    assign_faces_to_tiles, subset_mesh,
)

# Build a rotation matrix from Euler angles (radians).
rot = rotate_y(math.pi / 2) @ rotate_z(angle_z) @ rotate_x(angle_x)

# Bake a multi-part animated Model into a single world-space Mesh.
world_mesh = combine_model_world(meshes, model, frame=0)

# Assign each face to the nearest tile center (for large multi-tile scenery).
tile_ids = assign_faces_to_tiles(world_mesh, tile_centers_xz)

# Extract a per-tile sub-mesh (tightens scene bounds, improves AO accuracy).
tile_mesh = subset_mesh(world_mesh, tile_ids == 0)
```

### Multi-frame animation

`Model` and `MeshFrame` represent an animated object with up to four frames
(OpenRCT2's engine limit). Each placement holds one `MeshFrame` per frame:

```python
from openrct2_x7_renderer.types import Model, MeshFrame
import numpy as np

model = Model(meshes=[
    [
        MeshFrame(mesh_index=0, position=np.zeros(3),
                  orientation=np.array([0.0, 0.0, 0.0])),   # frame 0
        MeshFrame(mesh_index=0, position=np.zeros(3),
                  orientation=np.array([90.0, 0.0, 0.0])),  # frame 1
    ],
])
```

`orientation` is `(angle_y, angle_z, angle_x)` in degrees, applied as
`rotate_y @ rotate_z @ rotate_x`. `combine_model_world(meshes, model, frame=N)`
selects the pose for frame `N`; placements with fewer frames fall back to their
last frame.

### Config files

`parse_config` reads a JSON or YAML file into a plain dict (PyYAML is already
included as a package dependency). The `run_cli` helper builds the full context from a config
automatically, but you can load individual sections by hand:

```python
from openrct2_x7_renderer.config import parse_config
from openrct2_x7_renderer.lights import load_lights, default_lights

root = parse_config("object.json")   # or "object.yaml"
lights = load_lights(root["lights"]) if "lights" in root else default_lights()
```

A `lights` block in the config is a list of light objects:

```json
{
  "lights": [
    { "type": "diffuse",  "direction": [1.0, 1.65, -1.0], "strength": 0.8, "shadow": true },
    { "type": "specular", "direction": [0.0, 1.0,  0.0],  "strength": 0.5, "shadow": false },
    { "type": "hemi",     "direction": [0.0, -1.0, 0.0],  "strength": 0.1, "shadow": false }
  ]
}
```

`output_directory` (string) sets where generated files are written; omitting it
defaults to the current working directory.

`meshes` (array of strings) lists OBJ files to load. `preview` (string) points
to a preview PNG. Both accept absolute paths; relative paths are resolved
against the config file's parent directory when a `base_dir` is passed to
`load_meshes()` / `load_preview()` (the CLI does this automatically), falling
back to the current working directory when the file isn't found there.

#### Test-mode remap colours

`Remap1`/`Remap2`/`Remap3` materials render into reserved palette windows that
OpenRCT2 repaints to the player-chosen colour in-game, so they look wrong in a
static preview. An optional `test_remap_colors` block recolours those windows in
`--test` previews to show the repainted result. It is ignored outside test mode,
so real renders keep the raw remap windows OpenRCT2 expects.

```json
{
  "test_remap_colors": {
    "1": "bordeaux_red",
    "2": "dark_green",
    "3": "yellow"
  }
}
```

Keys are remap regions (`1`–`3`); values are OpenRCT2 colour names (the 32
standard colours, e.g. `grey`, `bright_red`, `teal`, `light_pink`). Any region
you omit keeps its raw remap window. `make_context(..., root=root)` reads the
block, and `Context.render_view` applies the substitution; the per-colour shade
ramps in `openrct2_x7_renderer.remap` are taken from OpenRCT2's own palette-map
sprites, so the preview matches the in-game repaint.

### Performance

The renderer maintains a persistent thread pool for the lifetime of each
`Context` — one worker thread per logical CPU core by default (capped at 256).
The pool is created when the `Context` is constructed and reused across all
render calls, so there is no thread-creation overhead on subsequent renders.
Set `OPENRCT2_X7_NUM_THREADS` to override the thread count:

```bash
OPENRCT2_X7_NUM_THREADS=4 python generate.py
```

## Sprite output format

`images_dat.write_images_dat` writes a single binary blob `images.dat`
containing all sprites, referenced from an OpenRCT2 `object.json` via the
`$LGX:` syntax (`"images": ["$LGX:images.dat[0..N-1]"]`). This is the same format
the vanilla OpenRCT2 parkobjs use.

### images.dat layout

```
+--------------------+--------------------+
| num_entries (u32)  | total_pixels (u32) |   8-byte header
+--------------------+--------------------+
| element 0          (16 bytes)           |
| ...                                     |   num_entries * 16 bytes
+-----------------------------------------+
| sprite 0 pixels    (w * h bytes)        |
| ...                                     |   total_pixels bytes
+-----------------------------------------+
```

Each element is `u32 offset, u16 width, u16 height, i16 x_offset,
i16 y_offset, u16 flags, u16 zoom`; width and height are unsigned. `flags = 0x0001` (`G1_FLAG_BMP`) indicates
raw indexed pixel data — palette index 0 is transparent. RLE compression
(`flags = 0x0008`) would be more compact but is not implemented.

## License

GPL-3.0-or-later. The distributed wheels bundle Embree and TBB (Apache-2.0);
their license texts ship alongside.
