/// Renderer.cpp

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "Mesh.hpp"
#include "Palette.hpp"
#include "Renderer.hpp"
#include "VectorMath.hpp"

namespace RCTGen {
    constexpr float kSqrt2 = std::numbers::sqrt2_v<float>;
    constexpr float kSqrt3 = std::numbers::sqrt3_v<float>;
    constexpr float kSqrt6 = std::numbers::sqrt2_v<float> * std::numbers::sqrt3_v<float>;

    constexpr int kAoSamplesU = 8;
    constexpr int kAoSamplesV = 4;
    constexpr int kAaSamplesU = 4;
    constexpr int kAaSamplesV = 4;
    constexpr float kAaSampleWeight = 1.0f / (kAaSamplesU * kAaSamplesV);
    // World-space depth tolerance used to classify samples as "inside" or "outside" an edge.
    constexpr float kEdgeDepthTolerance = 4.0f;
    // Fraction of the quantisation error propagated during Floyd-Steinberg dithering.
    // Standard F-S uses 1.0, but 0.3 significantly reduces colour-banding artifacts
    // in palette regions with coarse spacing (e.g. the remap and dark shadow regions).
    constexpr float kDitherErrorScale = 0.3f;

    // Screen-anchored ordered (Bayer) dithering. Unlike error diffusion the bias is
    // a pure function of the pixel's screen position, so it is identical across
    // animation frames wherever the colour is unchanged.
    constexpr int kBayerSize = 8;
    constexpr float kBayerLevels = static_cast<float>(kBayerSize * kBayerSize);
    // Amplitude of the dither perturbation, in 8-bit sRGB units. Roughly the spacing
    // between adjacent palette ramp entries; raise for more texture, lower for less.
    constexpr float kBayerSpread = 16.0f;
    // Standard recursive 8x8 Bayer threshold matrix, values in [0, 63].
    constexpr std::array<std::array<int, kBayerSize>, kBayerSize> kBayerMatrix = {{
        {{0, 32, 8, 40, 2, 34, 10, 42}},
        {{48, 16, 56, 24, 50, 18, 58, 26}},
        {{12, 44, 4, 36, 14, 46, 6, 38}},
        {{60, 28, 52, 20, 62, 30, 54, 22}},
        {{3, 35, 11, 43, 1, 33, 9, 41}},
        {{51, 19, 59, 27, 49, 17, 57, 25}},
        {{15, 47, 7, 39, 13, 45, 5, 37}},
        {{63, 31, 55, 23, 61, 29, 53, 21}},
    }};

    // Blank fragments used for bulk initialization
    constexpr Fragment kBlankFragment{
        {0.0f, 0.0f, 0.0f}, 0.0f, 0.0f, MaterialFlag::None, kFragmentUnused,
    };
    // Subsamples start with depth=∞ so any real hit is "closer".
    constexpr Fragment kBlankSample{
        {0.0f, 0.0f, 0.0f}, std::numeric_limits<float>::infinity(), 0.0f, MaterialFlag::None, kFragmentUnused,
    };

    // NOLINTNEXTLINE(misc-use-internal-linkage)
    void ContextInit(Context& ctx, std::span<const Light> lights, DitherMode dither, Palette palette, float upt) {
        ctx.rt_device = DeviceHandle::create();
        ctx.lights.assign(lights.begin(), lights.end());
        ctx.dither = dither;
        // Dimetric (2:1 pixel aspect, ~30° elevation) projection matrix.
        // Each entry is derived from (32/upt) * combinations of sqrt(2), sqrt(3), sqrt(6),
        // matching OpenRCT2's fixed isometric camera geometry.
        ctx.projection = {32.0f / upt,           0.0f,         -32.0f / upt,         -16.0f / upt,
                          -16.0f * kSqrt6 / upt, -16.0f / upt, 16.0f * kSqrt3 / upt, -16.0f * kSqrt2 / upt,
                          16.0f * kSqrt3 / upt};
        ctx.palette = palette;

        // Determine thread count once at context creation; reading the env var
        // inside the render hot path would re-scan the environment every row.
        constexpr unsigned int kMaxThreads = 256;
        unsigned int worker_count = std::min(std::max(1u, std::thread::hardware_concurrency()), kMaxThreads);
        if (const char* env = std::getenv("OPENRCT2_X7_NUM_THREADS")) {
            char* end = nullptr;
            const long n = std::strtol(env, &end, 10);
            if (end != env && n > 0) worker_count = std::min(static_cast<unsigned int>(n), kMaxThreads);
        }
        ctx.thread_pool = std::make_unique<ThreadPool>(worker_count);
    }

    void context_begin_render(Context& ctx) {
        ctx.finalized = false;
        ctx.rt_scene = std::make_unique<Scene>(ctx.rt_device.get());
    }

    void context_add_model(Context& ctx, const Mesh& mesh, Transform xform, MeshFlag mask) {
        SceneAddModel(
            *ctx.rt_scene, mesh,
            [xform](Vector3 v, Vector3 n) -> Vertex {
                return {transform_vector(xform, v), matrix_vector(xform.matrix, n).normalized()};
            },
            mask);
    }

    void context_finalize_render(Context& ctx) {
        ctx.rt_scene->finalize();
        ctx.finalized = true;
    }

    void context_end_render(Context& ctx) {
        ctx.rt_scene.reset();
    }

    void context_destroy(Context& ctx) {
        ctx.rt_device = {};
    }

    namespace {
        float Vector3DotClamped(Vector3 a, Vector3 b) {
            return std::max(vector3_dot(a, b), 0.0f);
        }

        Vector3 ShadeFragment(Scene& scene,
                              Vector3 pos,
                              Vector3 normal,
                              Vector3 view,
                              Vector3 color,
                              Vector3 specular_color,
                              float specular_exponent,
                              Vector3 ambient_color,
                              const std::vector<Light>& lights) {
            Vector3 output_color = vector3(0, 0, 0); // NOLINT(misc-const-correctness)

            for (const auto& light : lights) {
                if (light.shadow && scene_trace_occlusion_ray(scene, pos, light.direction)) continue;
                if (light.type == LightType::Hemi) {
                    float const diffuse_factor = 0.5f * light.intensity * (1 + vector3_dot(normal, light.direction));
                    output_color = vector3_add(vector3_mult(color, diffuse_factor), output_color);
                } else if (light.type == LightType::Diffuse) {
                    float const diffuse_factor = light.intensity * Vector3DotClamped(normal, light.direction);
                    output_color = vector3_add(vector3_mult(color, diffuse_factor), output_color);
                } else {
                    Vector3 const reflected_light_direction =
                        vector3_sub(vector3_mult(normal, 2.0f * vector3_dot(light.direction, normal)), light.direction);
                    float const specular_factor =
                        light.intensity
                        * std::pow(Vector3DotClamped(reflected_light_direction, view), specular_exponent);
                    output_color = vector3_add(vector3_mult(specular_color, specular_factor), output_color);
                }
            }
            return vector3_add(output_color, ambient_color);
        }

        // AO jitter is derived from a hash of the hit point so the same world
        // surface point produces the same (r1, r2) on every render
        inline std::uint32_t AoHashU32(std::uint32_t x) {
            x ^= x >> 17;
            x *= 0xed5ad4bbu;
            x ^= x >> 11;
            x *= 0xac4c1b51u;
            x ^= x >> 15;
            x *= 0x31848babu;
            x ^= x >> 14;
            return x;
        }

        inline std::uint32_t AoFloatBits(float f) {
            std::uint32_t b;
            std::memcpy(&b, &f, sizeof(b));
            return b;
        }

        inline float AoHashToUnit(std::uint32_t h) {
            return static_cast<float>(h >> 8) * (1.0f / 16777216.0f);
        }

        bool SceneSamplePoint(Scene& scene,
                              Vector2 point,
                              Matrix3 camera,
                              float ray_start_z,
                              const std::vector<Light>& lights,
                              Fragment& fragment) {
            RayHit hit{};
            Vector3 view_vector = matrix_vector(camera, vector3(0, 0, -1));
            if (scene_trace_ray(scene, matrix_vector(camera, vector3(point.x, point.y, ray_start_z)),
                                vector3_mult(view_vector, -1), hit)) {
                view_vector = vector3_normalize(view_vector);
                const Mesh* mesh = scene.meshes[hit.mesh_index];
                const Face& face = mesh->faces[hit.face_index];
                const Material& material = mesh->materials[face.material];

                // Check if this is a mask
                if (scene_is_mask(scene, static_cast<int>(hit.mesh_index))
                    || has_flag(material.flags, MaterialFlag::IsMask)) {
                    fragment.color = vector3(
                        0.0f, 0.0f, 0.0f); // transparent black; region=kFragmentUnused means pixel is not written
                    fragment.depth = hit.distance;
                    fragment.flags = material.flags | MaterialFlag::IsMask;
                    fragment.region = kFragmentUnused;
                    return true;
                }

                // Compute surface color
                Vector3 color;
                if (has_flag(material.flags, MaterialFlag::HasTexture)) {
                    Vector2 const tex_coord =
                        vector2_add(vector2_add(vector2_mult(mesh->uvs[face.indices[0]], 1.0f - hit.u - hit.v),
                                                vector2_mult(mesh->uvs[face.indices[1]], hit.u)),
                                    vector2_mult(mesh->uvs[face.indices[2]], hit.v));
                    color = texture_sample(material.texture, tex_coord);
                } else {
                    color = material.color;
                }
                // Remappable colors should be rendered as grayscale
                if (has_flag(material.flags, MaterialFlag::IsRemappable)) {
                    float const intensity = std::max({color.x, color.y, color.z});
                    color = vector3_from_scalar(intensity);
                }

                // Shade fragment
                Vector3 const shaded_color =
                    ShadeFragment(scene, hit.position, hit.normal, view_vector, color, material.specular_color,
                                  material.specular_exponent, material.ambient_color, lights);

                Vector3 const normal = hit.normal;
                // Guard against zero-length normals (degenerate geometry): the tangent frame
                // construction divides by sqrt(x²+z²) or sqrt(y²+z²), which is 0 for a
                // zero normal, producing NaN AO samples.  Skip AO instead (ao_factor = 1).
                const float normal_len_sq = normal.x * normal.x + normal.y * normal.y + normal.z * normal.z;

                float ao_factor = 1.0f;
                if (!has_flag(material.flags, MaterialFlag::NoAO) && normal_len_sq > 0.0f) {
                    Vector3 tangent; // NOLINT(misc-const-correctness)
                    if (std::fabs(normal.x) > std::fabs(normal.y))
                        tangent = vector3(normal.z, 0.0f, -normal.x)
                                  * (1.0f / std::sqrt(normal.x * normal.x + normal.z * normal.z));
                    else
                        tangent = vector3(0.0f, -normal.z, normal.y)
                                  * (1.0f / std::sqrt(normal.y * normal.y + normal.z * normal.z));
                    Vector3 const bitangent = vector3_cross(normal, tangent);
                    std::uint32_t hp = AoHashU32(AoFloatBits(hit.position.x));
                    hp = AoHashU32(hp ^ AoFloatBits(hit.position.y));
                    hp = AoHashU32(hp ^ AoFloatBits(hit.position.z));
                    std::uint32_t not_occluded_samples = 0;
                    for (int i = 0; i < kAoSamplesU; i++)
                        for (int j = 0; j < kAoSamplesV; j++) {
                            std::uint32_t const h = AoHashU32(hp ^ static_cast<std::uint32_t>(i * 73856093)
                                                              ^ static_cast<std::uint32_t>(j * 19349663));
                            float const r1 = AoHashToUnit(h);
                            float const r2 = AoHashToUnit(AoHashU32(h));
                            float const theta =
                                2.0f * std::numbers::pi_v<float> * ((static_cast<float>(i) + r1) / kAoSamplesU);
                            // acos(1-u) gives cos(phi) uniform in [0,1], i.e. uniform solid-angle sampling.
                            float const phi = std::acos(1.0f - (static_cast<float>(j) + r2) / kAoSamplesV);

                            Vector3 const local_sample_dir = vector3(std::cos(phi) * std::sin(theta),
                                                                     std::cos(phi) * std::cos(theta), std::sin(phi));
                            Vector3 const sample_dir =
                                vector3_add(vector3_mult(normal, local_sample_dir.z),
                                            vector3_add(vector3_mult(tangent, local_sample_dir.x),
                                                        vector3_mult(bitangent, local_sample_dir.y)));
                            if (!scene_trace_occlusion_ray(scene, hit.position, sample_dir)) not_occluded_samples++;
                        }
                    ao_factor = static_cast<float>(not_occluded_samples) / (kAoSamplesU * kAoSamplesV);
                }
                // Write result
                fragment.color = vector3_mult(shaded_color, ao_factor);
                fragment.depth = hit.distance;
                fragment.ghost_depth = hit.ghost_distance;
                fragment.flags = material.flags;
                fragment.region = material.region;
                return true;
            }
            fragment.ghost_depth = hit.ghost_distance;
            return false;
        }

        struct MaterialSample {
            float ghost_depth{};
            const Material* material{}; // nullptr on miss
            float depth{};
            bool is_mask{};
            explicit operator bool() const noexcept {
                return material != nullptr;
            }
        };

        MaterialSample SceneSampleMaterial(Scene& scene, Vector2 point, Matrix3 camera, float ray_start_z) {
            RayHit hit{};
            Vector3 const view_vector = matrix_vector(camera, vector3(0, 0, -1));
            if (scene_trace_ray(scene, matrix_vector(camera, vector3(point.x, point.y, ray_start_z)),
                                vector3_mult(view_vector, -1), hit)) {
                const Mesh* mesh = scene.meshes[hit.mesh_index];
                const Face& face = mesh->faces[hit.face_index];
                const Material* material = &mesh->materials[face.material];
                return {hit.ghost_distance, material, hit.distance,
                        scene_is_mask(scene, static_cast<int>(hit.mesh_index))
                            || has_flag(material->flags, MaterialFlag::IsMask)};
            }
            return {hit.ghost_distance};
        }

        Rect MakeRect(int x_lower, int x_upper, int y_lower, int y_upper) {
            return {x_lower, y_lower, x_upper, y_upper};
        }

        Rect RectEnclosePoint(Rect r, float x, float y) {
            return MakeRect(static_cast<int>(std::min(static_cast<float>(r.x_lower), std::floor(x))),
                            static_cast<int>(std::max(static_cast<float>(r.x_upper), std::ceil(x))),
                            static_cast<int>(std::min(static_cast<float>(r.y_lower), std::floor(y))),
                            static_cast<int>(std::max(static_cast<float>(r.y_upper), std::ceil(y))));
        }

        std::array<Vector3, 8> SceneBoundingCorners(const Scene& scene) {
            return {{
                vector3(scene.x_min, scene.y_min, scene.z_min),
                vector3(scene.x_max, scene.y_min, scene.z_min),
                vector3(scene.x_min, scene.y_max, scene.z_min),
                vector3(scene.x_max, scene.y_max, scene.z_min),
                vector3(scene.x_min, scene.y_min, scene.z_max),
                vector3(scene.x_max, scene.y_min, scene.z_max),
                vector3(scene.x_min, scene.y_max, scene.z_max),
                vector3(scene.x_max, scene.y_max, scene.z_max),
            }};
        }

        float SceneGetRayStartZ(Scene& scene, Matrix3 camera) {
            // Camera rays march from this screen-space depth towards the scene, so
            // it must lie in front of all geometry. Screen depth scales with
            // 1/upt, so a fixed constant gets overtaken once the scene is large or
            // the camera zoomed in; derive it from the projected bounding box.
            if (scene.x_min > scene.x_max) return -512.0f;
            float z_min = std::numeric_limits<float>::infinity();
            for (const Vector3& bp : SceneBoundingCorners(scene)) z_min = std::min(z_min, matrix_vector(camera, bp).z);
            return z_min - 2.0f;
        }

        Rect SceneGetBounds(Scene& scene, Matrix3 camera) {
            // An empty scene leaves the min/max extents at +/-infinity, and casting
            // those (or the NaNs they produce when projected) to int is undefined
            // behaviour.  Return the same margin a single point at the origin would
            // produce instead.
            if (scene.x_min > scene.x_max) return MakeRect(-1, 1, -1, 1);
            const std::array<Vector3, 8> bounding_points = SceneBoundingCorners(scene);

            Vector3 const first_screen = matrix_vector(camera, bounding_points[0]);
            Rect bounds =
                MakeRect(static_cast<int>(std::floor(first_screen.x)), static_cast<int>(std::ceil(first_screen.x)),
                         static_cast<int>(std::floor(first_screen.y)), static_cast<int>(std::ceil(first_screen.y)));
            for (const Vector3& bp : bounding_points) {
                Vector3 const screen_point = matrix_vector(camera, bp);
                bounds = RectEnclosePoint(bounds, screen_point.x, screen_point.y);
            }
            bounds.x_lower--;
            bounds.x_upper++;
            bounds.y_lower--;
            bounds.y_upper++;
            return bounds;
        }

        Rect FramebufferGetBounds(Framebuffer& framebuffer) {
            bool found_pixel = false;
            Rect bounds{};
            for (std::uint32_t y = 0; y < framebuffer.height; y++)
                for (std::uint32_t x = 0; x < framebuffer.width; x++) {
                    if (framebuffer.fragments[x + y * framebuffer.width].region != kFragmentUnused) {
                        if (found_pixel)
                            bounds = RectEnclosePoint(bounds, static_cast<float>(x), static_cast<float>(y));
                        else {
                            // Inclusive bounds, matching RectEnclosePoint: x_upper/y_upper
                            // are the last used pixel, not one past it.
                            bounds = MakeRect(static_cast<int>(x), static_cast<int>(x), static_cast<int>(y),
                                              static_cast<int>(y));
                            found_pixel = true;
                        }
                    }
                }
            if (!found_pixel)
                return MakeRect(0, 0, 0, 0);
            else
                return bounds;
        }

        Image ImageFromFramebuffer(Framebuffer& framebuffer, Palette& palette, DitherMode dither) {
            Image image;
            Rect const bounding_box = FramebufferGetBounds(framebuffer);
            image.width = static_cast<std::uint16_t>(1 + bounding_box.x_upper - bounding_box.x_lower);
            image.height = static_cast<std::uint16_t>(1 + bounding_box.y_upper - bounding_box.y_lower);
            // World origin (0, 0, 0) always projects to engine screen (0, 0)
            image.x_offset =
                static_cast<std::int16_t>(bounding_box.x_lower + static_cast<int>(std::floor(framebuffer.offset.x)));
            image.y_offset =
                static_cast<std::int16_t>(bounding_box.y_lower + static_cast<int>(std::floor(framebuffer.offset.y)));
            image.pixels.assign(static_cast<std::size_t>(image.width) * image.height, 0);

            // Engine-screen coordinate of framebuffer pixel (x, y): the bounding box
            // and offset both shift between animation frames, but their combination
            // (x + floor(offset)) is anchored to the world origin, giving the ordered
            // dither a frame-invariant phase.
            const int screen_origin_x = static_cast<int>(std::floor(framebuffer.offset.x));
            const int screen_origin_y = static_cast<int>(std::floor(framebuffer.offset.y));
            const auto clamp_u8 = [](float v) -> std::uint8_t {
                return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0f, 255.0f)));
            };

            for (int y = bounding_box.y_lower; y <= bounding_box.y_upper; y++) {
                int const start = (y & 1) ? (bounding_box.x_upper) : bounding_box.x_lower;
                int const stop = (y & 1) ? (bounding_box.x_lower - 1) : bounding_box.x_upper + 1;
                int const step = (y & 1) ? -1 : 1;

                for (int x = start; x != stop; x += step) {
                    Fragment& fragment = framebuffer.fragments[x + y * framebuffer.width];
                    if (fragment.region != kFragmentUnused) {
                        Color const base = color_from_vector(fragment.color);
                        // Snap to the 8-bit sRGB grid (kept for Floyd-Steinberg parity).
                        fragment.color = vector_from_color(base);

                        Vector3 target = fragment.color;
                        if (dither == DitherMode::Bayer) {
                            // A single per-pixel threshold shifts all channels together
                            // (avoids colour fringing); the index is the frame-invariant
                            // screen coordinate, so the pattern stays locked to the object.
                            const int sx = (x + screen_origin_x) & (kBayerSize - 1);
                            const int sy = (y + screen_origin_y) & (kBayerSize - 1);
                            const float threshold =
                                (static_cast<float>(kBayerMatrix[sy][sx]) + 0.5f) / kBayerLevels - 0.5f;
                            const float shift = threshold * kBayerSpread;
                            Color const perturbed = {clamp_u8(static_cast<float>(base.r) + shift),
                                                     clamp_u8(static_cast<float>(base.g) + shift),
                                                     clamp_u8(static_cast<float>(base.b) + shift)};
                            target = vector_from_color(perturbed);
                        }

                        PaletteResult const pr = PaletteGetNearest(palette, fragment.region & kRegionMask, target);
                        image.pixels[static_cast<std::size_t>(x - bounding_box.x_lower)
                                     + static_cast<std::size_t>(y - bounding_box.y_lower) * image.width] = pr.index;
                        if (dither == DitherMode::FloydSteinberg) {
                            // Distribute error onto neighbouring points (Floyd-Steinberg, serpentine)
                            const std::array<std::array<int, 2>, 4> points = {
                                {{x + step, y}, {x - step, y + 1}, {x, y + 1}, {x + step, y + 1}}};
                            constexpr std::array<float, 4> kWeights = {7.0f / 16.0f, 3.0f / 16.0f, 5.0f / 16.0f,
                                                                       1.0f / 16.0f};
                            for (int i = 0; i < 4; i++) {
                                const int px = points[i][0], py = points[i][1];
                                if (px >= 0 && std::cmp_less(px, framebuffer.width) && py >= 0
                                    && std::cmp_less(py, framebuffer.height)
                                    && (!has_flag(fragment.flags, MaterialFlag::NoBleed)
                                        || has_flag(framebuffer.fragments[px + py * framebuffer.width].flags,
                                                    MaterialFlag::NoBleed))) {
                                    framebuffer.fragments[px + py * framebuffer.width].color =
                                        vector3_add(vector3_mult(pr.error, kDitherErrorScale * kWeights[i]),
                                                    framebuffer.fragments[px + py * framebuffer.width].color);
                                }
                            }
                        }
                    }
                }
            }
            return image;
        }

        Image ContextRenderViewInternal(Context& ctx, Matrix3 view, bool silhouette) {
            assert(ctx.finalized && "context_render_view called before context_finalize_render");
            Matrix3 const camera = matrix_mult(ctx.projection, view);

            Rect const bounds = SceneGetBounds(*ctx.rt_scene, camera);
            float const ray_start_z = SceneGetRayStartZ(*ctx.rt_scene, camera);

            Framebuffer framebuffer;
            framebuffer.width = static_cast<std::uint32_t>(bounds.x_upper - bounds.x_lower + 1);
            framebuffer.height = static_cast<std::uint32_t>(bounds.y_upper - bounds.y_lower + 1);
            if (framebuffer.width > 16384 || framebuffer.height > 16384)
                throw std::runtime_error("Scene bounds too large to render (" + std::to_string(framebuffer.width) + "x"
                                         + std::to_string(framebuffer.height)
                                         + " pixels; max 16384x16384). Check units_per_tile or scene scale.");
            // Half-pixel shift on both axes
            framebuffer.offset =
                vector2(static_cast<float>(bounds.x_lower) - 0.5f, static_cast<float>(bounds.y_lower) - 0.5f);
            framebuffer.fragments.assign(static_cast<std::size_t>(framebuffer.width) * framebuffer.height,
                                         kBlankFragment);

            // Transform lights for view
            Matrix3 const view_inverse = matrix_inverse(view);
            std::vector<Light> transformed_lights;
            transformed_lights.reserve(ctx.lights.size());
            for (const Light& l : ctx.lights)
                transformed_lights.push_back({l.type, l.shadow, matrix_vector(view_inverse, l.direction), l.intensity});

            Matrix3 camera_inverse = matrix_inverse(camera);

            // Per-row work
            auto render_row = [&](int y) {
                for (int x = 0; std::cmp_less(x, framebuffer.width); x++) {
                    Vector2 const sample_point =
                        vector2_add(vector2(static_cast<float>(x), static_cast<float>(y)), framebuffer.offset);

                    // Test center
                    MaterialFlag flags = MaterialFlag::None;
                    int region = kFragmentUnused;
                    float depth = std::numeric_limits<float>::infinity();
                    bool mask = false;
                    auto center = SceneSampleMaterial(*ctx.rt_scene, sample_point, camera_inverse, ray_start_z);
                    float const ghost_depth = center.ghost_depth;
                    if (center) {
                        mask = center.is_mask;
                        region = mask ? kFragmentUnused : center.material->region;
                        flags = center.material->flags;
                        if (has_flag(center.material->flags, MaterialFlag::IsVisibleMask)) mask = true;
                        depth = center.depth;
                    }
                    // Compute subsamples
                    std::array<Fragment, static_cast<std::size_t>(kAaSamplesU) * kAaSamplesV> subsamples;
                    subsamples.fill(kBlankSample);
                    for (int i = 0; i < kAaSamplesU; i++)
                        for (int j = 0; j < kAaSamplesV; j++) {
                            const Vector2 subsample_point =
                                vector2((static_cast<float>(i) + 0.5f) / kAaSamplesU - 0.5f,
                                        (static_cast<float>(j) + 0.5f) / kAaSamplesV - 0.5f);
                            Fragment& sub_frag = subsamples[i + j * kAaSamplesU];

                            if (!silhouette) {
                                SceneSamplePoint(*ctx.rt_scene, vector2_add(sample_point, subsample_point),
                                                 camera_inverse, ray_start_z, transformed_lights, sub_frag);
                            } else {
                                auto sub =
                                    SceneSampleMaterial(*ctx.rt_scene, vector2_add(sample_point, subsample_point),
                                                        camera_inverse, ray_start_z);
                                sub_frag.ghost_depth = sub.ghost_depth;
                                if (sub) {
                                    sub_frag.color = vector3(0.5f, 0.5f, 0.5f);
                                    sub_frag.region = sub.is_mask ? kFragmentUnused : sub.material->region;
                                    sub_frag.flags = sub.material->flags;
                                    sub_frag.depth = sub.depth;
                                }
                            }
                        }

                    // Get frontmost background AA sample
                    int front_background_aa_sample = -1;
                    float min_depth = std::numeric_limits<float>::infinity();
                    for (int i = 0; i < kAaSamplesU * kAaSamplesV; i++) {
                        if (subsamples[i].depth < min_depth
                            && (has_flag(subsamples[i].flags, MaterialFlag::BackgroundAA)
                                || has_flag(subsamples[i].flags, MaterialFlag::BackgroundAADark))) {
                            front_background_aa_sample = i;
                            min_depth = subsamples[i].depth;
                        }
                    }

                    // If there exists a sample forward of the center point with background AA enabled,
                    // use that instead of the center point
                    if (front_background_aa_sample != -1 && (min_depth < ghost_depth - kEdgeDepthTolerance || mask)) {
                        // Count samples that fall inside the presumed edge
                        int inside_samples = 0;
                        for (int i = 0; i < kAaSamplesU * kAaSamplesV; i++) {
                            if (!(subsamples[i].depth > min_depth + kEdgeDepthTolerance
                                  || (subsamples[i].region == kFragmentUnused
                                      && !has_flag(subsamples[i].flags, MaterialFlag::IsMask))
                                  || has_flag(subsamples[i].flags, MaterialFlag::IsVisibleMask)))
                                inside_samples++;
                        }
                        // If more than three samples found, use the forwardmost point
                        if (inside_samples > 3) {
                            region = subsamples[front_background_aa_sample].region;
                            depth = min_depth;
                            flags = subsamples[front_background_aa_sample].flags;
                        }
                    }
                    framebuffer.fragments[x + y * framebuffer.width].region = static_cast<std::uint8_t>(region);
                    framebuffer.fragments[x + y * framebuffer.width].flags = flags;

                    // If this is a background pixel, there is no need to compute the color
                    if (std::cmp_equal(region, kFragmentUnused)) continue;

                    if (has_flag(flags, MaterialFlag::BackgroundAA)
                        || has_flag(flags, MaterialFlag::BackgroundAADark)) {
                        // Count samples that fall outside the presumed edge
                        Vector3 color = vector3(0, 0, 0);
                        float weight = 0;
                        float total_weight = 0;
                        for (int i = 0; i < kAaSamplesU * kAaSamplesV; i++) {
                            if ((!has_flag(subsamples[i].flags, MaterialFlag::NoBleed)
                                 || has_flag(flags, MaterialFlag::NoBleed))
                                && !((subsamples[i].ghost_depth <= depth + kEdgeDepthTolerance
                                      && subsamples[i].depth > depth + kEdgeDepthTolerance))) {
                                if (!(subsamples[i].depth > depth + kEdgeDepthTolerance
                                      || (subsamples[i].region == kFragmentUnused
                                          && !has_flag(subsamples[i].flags, MaterialFlag::IsMask))
                                      || has_flag(subsamples[i].flags, MaterialFlag::IsVisibleMask))) {
                                    color = vector3_add(color, vector3_mult(subsamples[i].color, kAaSampleWeight));
                                    weight += kAaSampleWeight;
                                }
                                total_weight += kAaSampleWeight;
                            }
                        }
                        if (total_weight > 0.0f) {
                            color = vector3_mult(color, 1.0f / total_weight);
                            if (has_flag(flags, MaterialFlag::BackgroundAADark))
                                framebuffer.fragments[x + y * framebuffer.width].color =
                                    vector3_mult(color, 0.5f + 0.5f * (weight / total_weight));
                            else
                                framebuffer.fragments[x + y * framebuffer.width].color = color;
                        }
                    } else {
                        Vector3 color = vector3(0, 0, 0);
                        float weight = 0.0f;
                        for (int i = 0; i < kAaSamplesU * kAaSamplesV; i++) {
                            if (subsamples[i].region != kFragmentUnused
                                && (!has_flag(subsamples[i].flags, MaterialFlag::NoBleed)
                                    || has_flag(flags, MaterialFlag::NoBleed))) {
                                color = vector3_add(color, vector3_mult(subsamples[i].color, kAaSampleWeight));
                                weight += kAaSampleWeight;
                            }
                        }
                        if (weight > 0.0f)
                            framebuffer.fragments[x + y * framebuffer.width].color = vector3_mult(color, 1.0f / weight);
                    }
                }
            };

            ctx.thread_pool->run(static_cast<int>(framebuffer.height), render_row);

            // Convert to indexed color
            return ImageFromFramebuffer(framebuffer, ctx.palette, ctx.dither);
        }
    } // namespace

    Image context_render_view(Context& ctx, Matrix3 view) {
        return ContextRenderViewInternal(ctx, view, /*silhouette=*/false);
    }

    Image context_render_silhouette(Context& ctx, Matrix3 view) {
        return ContextRenderViewInternal(ctx, view, /*silhouette=*/true);
    }
} // namespace RCTGen
