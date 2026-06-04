# OpenRCT2 X7 Renderer

`openrct2-x7-renderer` is the Embree-backed isometric renderer that turns
triangle meshes into OpenRCT2 palette-indexed sprites.

Heavily inspired by X7's [RCTGen](https://github.com/X123M3-256/RCTGen).

## Install

Embree-vendored wheels are published to PyPI:

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
uv run pytest
```

## Quick start

```python
import numpy as np

from openrct2_x7_renderer.mesh import load_mesh
from openrct2_x7_renderer.lights import default_lights
from openrct2_x7_renderer.ray_trace import Context, render_view, VIEWS

mesh = load_mesh("model.obj")
ctx = Context.make(lights=default_lights(), dither=True)
ctx.begin_render()
ctx.add_model(mesh, matrix=np.eye(3), translation=np.zeros(3))
ctx.finalize_render()
sprite = render_view(ctx, VIEWS[0])   # -> IndexedImage
ctx.end_render()
```

## Sprite output format

`images_dat.write_images_dat` writes a single binary blob `images.dat`
containing all sprites, referenced from an OpenRCT2 `object.json` via the
`$LGX:` syntax (`"images": ["$LGX:images.dat[0..N]"]`). This is the same format
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

Each element is `u32 offset, i16 width, i16 height, i16 x_offset,
i16 y_offset, u16 flags, u16 zoom`. `flags = 0x0001` (`G1_FLAG_BMP`) indicates
raw indexed pixel data — palette index 0 is transparent. RLE compression
(`flags = 0x0008`) would be more compact but is not implemented.
