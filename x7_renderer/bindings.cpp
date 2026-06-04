/// bindings.cpp

// pybind11 surface for OpenRCT2-X7-Renderer.
//
// Exposes the Embree-backed renderer kernel.
// Python builds scenes from numpy arrays, the C++ side does ray tracing.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "Mesh.hpp"
#include "Palette.hpp"
#include "RayTrace.hpp"
#include "Renderer.hpp"
#include "VectorMath.hpp"

namespace py = pybind11;

namespace {
    using namespace RCTGen;

    Matrix3 Matrix3FromArray(const py::array_t<float, py::array::c_style | py::array::forcecast>& arr) {
        if (arr.ndim() != 2 || arr.shape(0) != 3 || arr.shape(1) != 3)
            throw std::invalid_argument("matrix must be a (3, 3) float array");
        Matrix3 m{};
        auto r = arr.unchecked<2>();
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) m(i, j) = r(i, j);
        return m;
    }

    Vector3 Vec3FromArray(const py::array_t<float, py::array::c_style | py::array::forcecast>& arr) {
        if (arr.ndim() != 1 || arr.shape(0) != 3) throw std::invalid_argument("vector must be a (3,) float array");
        auto r = arr.unchecked<1>();
        return vector3(r(0), r(1), r(2));
    }

    Vector3 Vec3FromSeq(const py::sequence& s) {
        if (py::len(s) != 3) throw std::invalid_argument("vector must have 3 components");
        return vector3(s[0].cast<float>(), s[1].cast<float>(), s[2].cast<float>());
    }

    struct OwnedMesh {
        Mesh mesh{};
        std::vector<Vector3> vertices;
        std::vector<Vector3> normals;
        std::vector<Vector2> uvs;
        std::vector<Face> faces;
        std::vector<Material> materials;
        std::vector<std::vector<Vector3>> texture_pixels;
    };

    class RenderContext {
    public:
        RenderContext(const py::list& lights_py, bool dither, float upt) {
            lights_.reserve(py::len(lights_py));
            for (auto h : lights_py) {
                auto const o = py::reinterpret_borrow<py::object>(h);
                Light light{};
                light.type = static_cast<LightType>(o.attr("type").cast<std::uint16_t>());
                light.shadow = o.attr("shadow").cast<bool>() ? 1 : 0;
                light.intensity = o.attr("intensity").cast<float>();
                auto const dir =
                    o.attr("direction").cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
                light.direction = Vec3FromArray(dir);
                lights_.push_back(light);
            }

            palette_ = palette_rct2();
            ContextInit(ctx_, std::span<const Light>(lights_), dither, palette_, upt);
        }

        ~RenderContext() {
            // The user is supposed to call end_render() but be defensive.
            if (scene_open_) {
                context_end_render(ctx_);
                scene_open_ = false;
            }
            context_destroy(ctx_);
        }

        RenderContext(const RenderContext&) = delete;
        RenderContext& operator=(const RenderContext&) = delete;

        void BeginRender() {
            if (scene_open_) throw std::runtime_error("begin_render called twice without end_render");
            context_begin_render(ctx_);
            owned_meshes_.clear();
            scene_open_ = true;
            finalized_ = false;
        }

        void EndRender() {
            if (!scene_open_) return;
            context_end_render(ctx_);
            owned_meshes_.clear();
            scene_open_ = false;
            finalized_ = false;
        }

        void FinalizeRender() {
            if (!scene_open_) throw std::runtime_error("finalize_render before begin_render");
            context_finalize_render(ctx_);
            finalized_ = true;
        }

        void AddMesh(const py::array_t<float, py::array::c_style | py::array::forcecast>& vertices,
                     const py::array_t<float, py::array::c_style | py::array::forcecast>& normals,
                     const py::array_t<float, py::array::c_style | py::array::forcecast>& uvs,
                     const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>& faces,
                     const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>& face_materials,
                     const py::list& materials,
                     const py::array_t<float, py::array::c_style | py::array::forcecast>& matrix,
                     const py::array_t<float, py::array::c_style | py::array::forcecast>& translation,
                     int mask) {
            if (!scene_open_) throw std::runtime_error("add_mesh before begin_render");
            if (finalized_) throw std::runtime_error("add_mesh after finalize_render");

            // Shape checks.
            if (vertices.ndim() != 2 || vertices.shape(1) != 3) throw std::invalid_argument("vertices must be (N, 3)");
            if (normals.ndim() != 2 || normals.shape(1) != 3 || normals.shape(0) != vertices.shape(0))
                throw std::invalid_argument("normals must be (N, 3) matching vertices");
            if (uvs.ndim() != 2 || uvs.shape(1) != 2 || uvs.shape(0) != vertices.shape(0))
                throw std::invalid_argument("uvs must be (N, 2) matching vertices");
            if (faces.ndim() != 2 || faces.shape(1) != 3) throw std::invalid_argument("faces must be (F, 3)");
            if (face_materials.ndim() != 1 || face_materials.shape(0) != faces.shape(0))
                throw std::invalid_argument("face_materials must be (F,) matching faces");

            const auto num_vertices = vertices.shape(0);
            const auto num_faces = faces.shape(0);
            const auto num_materials = static_cast<std::size_t>(py::len(materials));

            // Allocate owned slot in-place to keep pointer addresses stable.
            owned_meshes_.push_back(std::make_unique<OwnedMesh>());
            OwnedMesh* om = owned_meshes_.back().get();

            om->vertices.resize(static_cast<std::size_t>(num_vertices));
            om->normals.resize(static_cast<std::size_t>(num_vertices));
            om->uvs.resize(static_cast<std::size_t>(num_vertices));
            om->faces.resize(static_cast<std::size_t>(num_faces));

            auto v = vertices.unchecked<2>();
            auto n = normals.unchecked<2>();
            auto u = uvs.unchecked<2>();
            for (py::ssize_t i = 0; i < num_vertices; i++) {
                om->vertices[i] = vector3(v(i, 0), v(i, 1), v(i, 2));
                om->normals[i] = vector3(n(i, 0), n(i, 1), n(i, 2));
                om->uvs[i] = vector2(u(i, 0), u(i, 1));
            }

            auto fi = faces.unchecked<2>();
            auto fm = face_materials.unchecked<1>();
            for (py::ssize_t i = 0; i < num_faces; i++) {
                if (static_cast<std::size_t>(fm(i)) >= num_materials)
                    throw std::invalid_argument("face_materials[" + std::to_string(i) + "] = " + std::to_string(fm(i))
                                                + " is out of range (num materials = " + std::to_string(num_materials)
                                                + ")");
                if (std::cmp_greater_equal(fi(i, 0), num_vertices) || std::cmp_greater_equal(fi(i, 1), num_vertices)
                    || std::cmp_greater_equal(fi(i, 2), num_vertices))
                    throw std::invalid_argument("face[" + std::to_string(i)
                                                + "] has a vertex index out of range (num vertices = "
                                                + std::to_string(num_vertices) + ")");
                om->faces[i].material = static_cast<std::size_t>(fm(i));
                om->faces[i].indices[0] = static_cast<std::size_t>(fi(i, 0));
                om->faces[i].indices[1] = static_cast<std::size_t>(fi(i, 1));
                om->faces[i].indices[2] = static_cast<std::size_t>(fi(i, 2));
            }

            // Materials.
            om->materials.resize(num_materials);
            om->texture_pixels.resize(num_materials);
            for (std::size_t i = 0; i < num_materials; i++) {
                auto const d = materials[i].cast<py::dict>();
                Material& mat = om->materials[i];
                mat.flags = static_cast<MaterialFlag>(d["flags"].cast<std::uint16_t>());
                mat.region = d["region"].cast<std::uint8_t>();
                mat.specular_exponent = d["specular_exponent"].cast<float>();
                mat.specular_color = Vec3FromSeq(d["specular_color"].cast<py::sequence>());
                mat.ambient_color = Vec3FromSeq(d["ambient_color"].cast<py::sequence>());

                if (has_flag(mat.flags, MaterialFlag::HasTexture)) {
                    auto const tex = d["texture"].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
                    if (tex.ndim() != 3 || tex.shape(2) != 3)
                        throw std::invalid_argument("texture must be (H, W, 3) float32 linear RGB");
                    const auto tex_h = tex.shape(0);
                    const auto tex_w = tex.shape(1);
                    auto& buf = om->texture_pixels[i];
                    buf.resize(static_cast<std::size_t>(tex_h) * static_cast<std::size_t>(tex_w));
                    auto t = tex.unchecked<3>();
                    for (py::ssize_t y = 0; y < tex_h; y++)
                        for (py::ssize_t x = 0; x < tex_w; x++)
                            buf[y * tex_w + x] = vector3(t(y, x, 0), t(y, x, 1), t(y, x, 2));
                    mat.texture.width = static_cast<std::uint16_t>(tex_w);
                    mat.texture.height = static_cast<std::uint16_t>(tex_h);
                    mat.texture.pixels = std::span<const Vector3>(buf);
                } else {
                    mat.color = Vec3FromSeq(d["color"].cast<py::sequence>());
                }
            }

            om->mesh.vertices = std::span<const Vector3>(om->vertices);
            om->mesh.normals = std::span<const Vector3>(om->normals);
            om->mesh.uvs = std::span<const Vector2>(om->uvs);
            om->mesh.faces = std::span<const Face>(om->faces);
            om->mesh.materials = std::span<const Material>(om->materials);

            // Apply transform.
            Transform const xform = transform(Matrix3FromArray(matrix), Vec3FromArray(translation));
            context_add_model(ctx_, om->mesh, xform, static_cast<MeshFlag>(mask));
        }

        py::dict RenderView(const py::array_t<float, py::array::c_style | py::array::forcecast>& view) {
            return RenderInternal(view, /*silhouette=*/false);
        }

        py::dict RenderSilhouette(const py::array_t<float, py::array::c_style | py::array::forcecast>& view) {
            return RenderInternal(view, /*silhouette=*/true);
        }

    private:
        py::dict RenderInternal(const py::array_t<float, py::array::c_style | py::array::forcecast>& view,
                                bool silhouette) {
            if (!finalized_) throw std::runtime_error("render_view/render_silhouette before finalize_render");
            Matrix3 const m = Matrix3FromArray(view);
            // Release the GIL for the ray-tracing hot path so a Python worker
            // thread keeps its host UI responsive while a render is in flight
            Image img;
            {
                py::gil_scoped_release const release;
                img = silhouette ? context_render_silhouette(ctx_, m) : context_render_view(ctx_, m);
            }

            // Copy pixels into a numpy array.
            const py::ssize_t img_h = img.height;
            const py::ssize_t img_w = img.width;
            py::array_t<std::uint8_t> pixels({img_h, img_w});
            if (!img.pixels.empty() && img_h > 0 && img_w > 0) {
                std::memcpy(pixels.mutable_data(), img.pixels.data(), img.pixels.size());
            }

            py::dict d;
            d["width"] = static_cast<int>(img.width);
            d["height"] = static_cast<int>(img.height);
            d["x_offset"] = static_cast<int>(img.x_offset);
            d["y_offset"] = static_cast<int>(img.y_offset);
            d["pixels"] = pixels;
            return d;
        }

        Context ctx_{};
        Palette palette_{};
        std::vector<Light> lights_;
        std::vector<std::unique_ptr<OwnedMesh>> owned_meshes_;
        bool scene_open_ = false;
        bool finalized_ = false;
    };

} // namespace

namespace {
    py::array_t<std::uint8_t> GetPaletteRgb() {
        using namespace RCTGen;
        Palette const pal = palette_rct2();
        py::array_t<std::uint8_t> out({256, 3});
        auto buf = out.mutable_unchecked<2>();
        for (int i = 0; i < 256; ++i) {
            buf(i, 0) = pal.colors[i].r;
            buf(i, 1) = pal.colors[i].g;
            buf(i, 2) = pal.colors[i].b;
        }
        return out;
    }
} // namespace

PYBIND11_MODULE(_x7_renderer, m) {
    using namespace RCTGen;
    m.doc() = "Native renderer (Embree) for OpenRCT2-X7-Renderer.";

    m.def("palette_rgb", &GetPaletteRgb, "Return the native RCT2 palette as a (256, 3) uint8 array.");

    // Re-export constants the Python wrapper uses to construct material
    // dicts.
    m.attr("MATERIAL_HAS_TEXTURE") = static_cast<int>(MaterialFlag::HasTexture);
    m.attr("MATERIAL_IS_REMAPPABLE") = static_cast<int>(MaterialFlag::IsRemappable);
    m.attr("MATERIAL_IS_MASK") = static_cast<int>(MaterialFlag::IsMask);
    m.attr("MATERIAL_NO_AO") = static_cast<int>(MaterialFlag::NoAO);
    m.attr("MATERIAL_BACKGROUND_AA") = static_cast<int>(MaterialFlag::BackgroundAA);
    m.attr("MATERIAL_BACKGROUND_AA_DARK") = static_cast<int>(MaterialFlag::BackgroundAADark);
    m.attr("MATERIAL_IS_VISIBLE_MASK") = static_cast<int>(MaterialFlag::IsVisibleMask);
    m.attr("MATERIAL_NO_BLEED") = static_cast<int>(MaterialFlag::NoBleed);
    m.attr("MATERIAL_IS_FLAT_SHADED") = static_cast<int>(MaterialFlag::IsFlatShaded);
    m.attr("MESH_MASK") = static_cast<int>(MeshFlag::Mask);
    m.attr("MESH_GHOST") = static_cast<int>(MeshFlag::Ghost);
    m.attr("LIGHT_HEMI") = static_cast<int>(LightType::Hemi);
    m.attr("LIGHT_DIFFUSE") = static_cast<int>(LightType::Diffuse);
    m.attr("LIGHT_SPECULAR") = static_cast<int>(LightType::Specular);

    py::class_<RenderContext>(m, "Context")
        .def(py::init<py::list, bool, float>(), py::arg("lights"), py::arg("dither"), py::arg("upt"))
        .def("begin_render", &RenderContext::BeginRender)
        .def("add_mesh", &RenderContext::AddMesh, py::arg("vertices"), py::arg("normals"), py::arg("uvs"),
             py::arg("faces"), py::arg("face_materials"), py::arg("materials"), py::arg("matrix"),
             py::arg("translation"), py::arg("mask") = 0)
        .def("finalize_render", &RenderContext::FinalizeRender)
        .def("render_view", &RenderContext::RenderView, py::arg("view"))
        .def("render_silhouette", &RenderContext::RenderSilhouette, py::arg("view"))
        .def("end_render", &RenderContext::EndRender);
}
