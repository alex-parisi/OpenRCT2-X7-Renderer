/// Renderer.cpp

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numbers>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include "Mesh.hpp"
#include "Palette.hpp"
#include "Renderer.hpp"
#include "VectorMath.hpp"

namespace RCTGen {
    // 3.67 metres per tile
    constexpr float kSqrt2 = std::numbers::sqrt2_v<float>;
    constexpr float kSqrt3 = std::numbers::sqrt3_v<float>;
    constexpr float kSqrt6 = std::numbers::sqrt2_v<float> * std::numbers::sqrt3_v<float>;

    constexpr int kAoSamplesU = 8;
    constexpr int kAoSamplesV = 4;
    constexpr int kAaSamplesU = 4;
    constexpr int kAaSamplesV = 4;
    constexpr float kAaSampleWeight = 1.0f / (kAaSamplesU * kAaSamplesV);

    // Blank fragments used for bulk initialization
    constexpr Fragment kBlankFragment{
        {0.0f, 0.0f, 0.0f}, 0.0f, 0.0f, 0, kFragmentUnused,
    };
    // Subsamples start with depth=∞ so any real hit is "closer".
    constexpr Fragment kBlankSample{
        {0.0f, 0.0f, 0.0f}, std::numeric_limits<float>::infinity(), 0.0f, 0, kFragmentUnused,
    };

    std::array<Matrix3, 4> views{{
        {{1, 0, 0, 0, 1, 0, 0, 0, 1}},
        {{0, 0, 1, 0, 1, 0, -1, 0, 0}},
        {{-1, 0, 0, 0, 1, 0, 0, 0, -1}},
        {{0, 0, -1, 0, 1, 0, 1, 0, 0}},
    }};

    namespace {
        // Spawn a batch of worker threads, dispatch `count` units of work,
        // join.
        template <class Fn> void parallel_for(int count, Fn&& fn) {
            if (count <= 0) return;
            const unsigned int worker_count = std::max(1u, std::thread::hardware_concurrency());
            std::atomic<int> next{0};
            auto worker = [&]() {
                for (;;) {
                    int const i = next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= count) break;
                    fn(i);
                }
            };
            std::vector<std::thread> threads;
            threads.reserve(worker_count);
            for (unsigned int i = 0; i < worker_count; ++i) threads.emplace_back(worker);
            for (auto& t : threads) t.join();
        }
    } // namespace

    void context_init(Context& ctx, std::span<const Light> lights, bool dither, Palette palette, float upt) {
        ctx.rt_device = device_init();
        ctx.lights.assign(lights.begin(), lights.end());
        ctx.dither = dither;
        // Dimetric projection
        ctx.projection = {32.0f / upt,           0.0f,         -32.0f / upt,         -16.0f / upt,
                          -16.0f * kSqrt6 / upt, -16.0f / upt, 16.0f * kSqrt3 / upt, -16.0f * kSqrt2 / upt,
                          16.0f * kSqrt3 / upt};
        ctx.palette = palette;
    }

    void context_begin_render(Context& ctx) { scene_init(ctx.rt_scene, ctx.rt_device); }

    void context_add_model(Context& ctx, const Mesh& mesh, Transform xform, int mask) {
        scene_add_model(
            ctx.rt_scene, mesh,
            [xform](Vector3 v, Vector3 n, bool /*flat_shaded*/) -> Vertex {
                return {transform_vector(xform, v), matrix_vector(xform.matrix, n).normalized()};
            },
            mask);
    }

    void context_finalize_render(Context& ctx) { scene_finalize(ctx.rt_scene); }

    void context_end_render(Context& ctx) { scene_destroy(ctx.rt_scene); }

    void context_destroy(Context& ctx) { device_destroy(ctx.rt_device); }

    namespace {
        float vector3_dot_clamped(Vector3 a, Vector3 b) { return std::max(vector3_dot(a, b), 0.0f); }

        Vector3 shade_fragment(Scene& scene,
                               Vector3 pos,
                               Vector3 normal,
                               Vector3 view,
                               Vector3 color,
                               Vector3 specular_color,
                               float specular_exponent,
                               Vector3 ambient_color,
                               const std::vector<Light>& lights) {
            Vector3 output_color = vector3(0, 0, 0);

            for (const auto& light : lights) {
                if (light.shadow && scene_trace_occlusion_ray(scene, pos, light.direction)) continue;
                if (light.type == LIGHT_HEMI) {
                    float const diffuse_factor = 0.5f * light.intensity * (1 + vector3_dot(normal, light.direction));
                    output_color = vector3_add(vector3_mult(color, diffuse_factor), output_color);
                } else if (light.type == LIGHT_DIFFUSE) {
                    float const diffuse_factor = light.intensity * vector3_dot_clamped(normal, light.direction);
                    output_color = vector3_add(vector3_mult(color, diffuse_factor), output_color);
                } else {
                    Vector3 const reflected_light_direction =
                        vector3_sub(vector3_mult(normal, 2.0f * vector3_dot(light.direction, normal)), light.direction);
                    float const specular_factor =
                        light.intensity
                        * std::pow(vector3_dot_clamped(reflected_light_direction, view), specular_exponent);
                    output_color = vector3_add(vector3_mult(specular_color, specular_factor), output_color);
                }
            }
            return vector3_add(output_color, ambient_color);
        }

        // AO jitter is derived from a hash of the hit point so the same world
        // surface point produces the same (r1, r2) on every render
        inline std::uint32_t ao_hash_u32(std::uint32_t x) {
            x ^= x >> 17;
            x *= 0xed5ad4bbu;
            x ^= x >> 11;
            x *= 0xac4c1b51u;
            x ^= x >> 15;
            x *= 0x31848babu;
            x ^= x >> 14;
            return x;
        }

        inline std::uint32_t ao_float_bits(float f) {
            std::uint32_t b;
            std::memcpy(&b, &f, sizeof(b));
            return b;
        }

        inline float ao_hash_to_unit(std::uint32_t h) { return static_cast<float>(h >> 8) * (1.0f / 16777216.0f); }

        bool scene_sample_point(
            Scene& scene, Vector2 point, Matrix3 camera, const std::vector<Light>& lights, Fragment& fragment) {
            RayHit hit{};
            Vector3 view_vector = matrix_vector(camera, vector3(0, 0, -1));
            if (scene_trace_ray(scene, matrix_vector(camera, vector3(point.x, point.y, -512)),
                                vector3_mult(view_vector, -1), hit)) {
                view_vector = vector3_normalize(view_vector);
                const Mesh* mesh = scene.meshes[hit.mesh_index];
                const Face& face = mesh->faces[hit.face_index];
                const Material& material = mesh->materials[face.material];

                // Check if this is a mask
                if (scene_is_mask(scene, static_cast<int>(hit.mesh_index)) || material.flags & MATERIAL_IS_MASK) {
                    fragment.color = vector3(0, 1, 0);
                    fragment.depth = hit.distance;
                    fragment.flags = static_cast<std::uint8_t>(material.flags | MATERIAL_IS_MASK);
                    fragment.region = kFragmentUnused;
                    return true;
                }

                // Compute surface color
                Vector3 color;
                if (material.flags & MATERIAL_HAS_TEXTURE) {
                    Vector2 const tex_coord =
                        vector2_add(vector2_add(vector2_mult(mesh->uvs[face.indices[0]], 1.0f - hit.u - hit.v),
                                                vector2_mult(mesh->uvs[face.indices[1]], hit.u)),
                                    vector2_mult(mesh->uvs[face.indices[2]], hit.v));
                    color = texture_sample(material.texture, tex_coord);
                } else {
                    color = material.color;
                }
                // Remappable colors should be rendered as grayscale
                if (material.flags & MATERIAL_IS_REMAPPABLE) {
                    float const intensity = std::max({color.x, color.y, color.z});
                    color = vector3_from_scalar(intensity);
                }

                // Shade fragment
                Vector3 const shaded_color =
                    shade_fragment(scene, hit.position, hit.normal, view_vector, color, material.specular_color,
                                   material.specular_exponent, material.ambient_color, lights);

                Vector3 const normal = hit.normal;
                Vector3 tangent;
                if (std::fabs(normal.x) > std::fabs(normal.y))
                    tangent = vector3(normal.z, 0.0f, -normal.x)
                              * (1.0f / std::sqrt(normal.x * normal.x + normal.z * normal.z));
                else
                    tangent = vector3(0.0f, -normal.z, normal.y)
                              * (1.0f / std::sqrt(normal.y * normal.y + normal.z * normal.z));
                Vector3 const bitangent = vector3_cross(normal, tangent);

                float ao_factor = 1.0f;
                if (!(material.flags & MATERIAL_NO_AO)) {
                    std::uint32_t hp = ao_hash_u32(ao_float_bits(hit.position.x));
                    hp = ao_hash_u32(hp ^ ao_float_bits(hit.position.y));
                    hp = ao_hash_u32(hp ^ ao_float_bits(hit.position.z));
                    std::uint32_t not_occluded_samples = 0;
                    for (int i = 0; i < kAoSamplesU; i++)
                        for (int j = 0; j < kAoSamplesV; j++) {
                            std::uint32_t const h = ao_hash_u32(hp ^ static_cast<std::uint32_t>(i * 73856093)
                                                                ^ static_cast<std::uint32_t>(j * 19349663));
                            float const r1 = ao_hash_to_unit(h);
                            float const r2 = ao_hash_to_unit(ao_hash_u32(h));
                            float const theta =
                                2.0f * std::numbers::pi_v<float> * ((static_cast<float>(i) + r1) / kAoSamplesU);
                            float const phi = std::asin(1.0f - (static_cast<float>(j) + r2) / kAoSamplesV);

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
                fragment.flags = static_cast<std::uint8_t>(material.flags);
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
            explicit operator bool() const noexcept { return material != nullptr; }
        };

        MaterialSample scene_sample_material(Scene& scene, Vector2 point, Matrix3 camera) {
            RayHit hit{};
            Vector3 const view_vector = matrix_vector(camera, vector3(0, 0, -1));
            if (scene_trace_ray(scene, matrix_vector(camera, vector3(point.x, point.y, -512)),
                                vector3_mult(view_vector, -1), hit)) {
                const Mesh* mesh = scene.meshes[hit.mesh_index];
                const Face& face = mesh->faces[hit.face_index];
                const Material* material = &mesh->materials[face.material];
                return {hit.ghost_distance, material, hit.distance,
                        scene_is_mask(scene, static_cast<int>(hit.mesh_index))
                            || static_cast<bool>(material->flags & MATERIAL_IS_MASK)};
            }
            return {hit.ghost_distance};
        }

        Rect rect(int xl, int xu, int yl, int yu) {
            Rect result = {xl, yl, xu, yu};
            return result;
        }

        Rect rect_enclose_point(Rect r, float x, float y) {
            return rect(static_cast<int>(std::min(static_cast<float>(r.x_lower), std::floor(x))),
                        static_cast<int>(std::max(static_cast<float>(r.x_upper), std::ceil(x))),
                        static_cast<int>(std::min(static_cast<float>(r.y_lower), std::floor(y))),
                        static_cast<int>(std::max(static_cast<float>(r.y_upper), std::ceil(y))));
        }

        Rect scene_get_bounds(Scene& scene, Matrix3 camera) {
            const std::array<Vector3, 8> bounding_points = {{
                vector3(scene.x_min, scene.y_min, scene.z_min),
                vector3(scene.x_max, scene.y_min, scene.z_min),
                vector3(scene.x_min, scene.y_max, scene.z_min),
                vector3(scene.x_max, scene.y_max, scene.z_min),
                vector3(scene.x_min, scene.y_min, scene.z_max),
                vector3(scene.x_max, scene.y_min, scene.z_max),
                vector3(scene.x_min, scene.y_max, scene.z_max),
                vector3(scene.x_max, scene.y_max, scene.z_max),
            }};

            Rect bounds = rect(
                static_cast<int>(std::floor(bounding_points[0].x)), static_cast<int>(std::ceil(bounding_points[0].x)),
                static_cast<int>(std::floor(bounding_points[0].y)), static_cast<int>(std::ceil(bounding_points[0].y)));
            for (const Vector3& bp : bounding_points) {
                Vector3 const screen_point = matrix_vector(camera, bp);
                bounds = rect_enclose_point(bounds, screen_point.x, screen_point.y);
            }
            bounds.x_lower--;
            bounds.x_upper++;
            bounds.y_lower--;
            bounds.y_upper++;
            return bounds;
        }

        Rect framebuffer_get_bounds(Framebuffer& framebuffer) {
            bool found_pixel = false;
            Rect bounds{};
            for (std::uint32_t y = 0; y < framebuffer.height; y++)
                for (std::uint32_t x = 0; x < framebuffer.width; x++) {
                    if (framebuffer.fragments[x + y * framebuffer.width].region != kFragmentUnused) {
                        if (found_pixel)
                            bounds = rect_enclose_point(bounds, static_cast<float>(x), static_cast<float>(y));
                        else {
                            bounds = rect(static_cast<int>(x), static_cast<int>(x) + 1, static_cast<int>(y),
                                          static_cast<int>(y) + 1);
                            found_pixel = true;
                        }
                    }
                }
            if (!found_pixel)
                return rect(0, 0, 0, 0);
            else
                return bounds;
        }

        Image image_from_framebuffer(Framebuffer& framebuffer, Palette& palette, bool dither) {
            Image image;
            Rect const bounding_box = framebuffer_get_bounds(framebuffer);
            image.width = static_cast<std::uint16_t>(1 + bounding_box.x_upper - bounding_box.x_lower);
            image.height = static_cast<std::uint16_t>(1 + bounding_box.y_upper - bounding_box.y_lower);
            // World origin (0, 0, 0) always projects to engine screen (0, 0)
            image.x_offset =
                static_cast<std::int16_t>(bounding_box.x_lower + static_cast<int>(std::floor(framebuffer.offset.x)));
            image.y_offset =
                static_cast<std::int16_t>(bounding_box.y_lower + static_cast<int>(std::floor(framebuffer.offset.y)));
            image.pixels.assign(static_cast<std::size_t>(image.width) * image.height, 0);

            for (int y = bounding_box.y_lower; y <= bounding_box.y_upper; y++) {
                int const start = (y & 1) ? (bounding_box.x_upper) : bounding_box.x_lower;
                int const stop = (y & 1) ? (bounding_box.x_lower - 1) : bounding_box.x_upper + 1;
                int const step = (y & 1) ? -1 : 1;

                for (int x = start; x != stop; x += step) {
                    Fragment& fragment = framebuffer.fragments[x + y * framebuffer.width];
                    fragment.color = vector_from_color(color_from_vector(fragment.color));
                    if (fragment.region != kFragmentUnused) {
                        PaletteResult const pr =
                            palette_get_nearest(palette, fragment.region & kRegionMask, fragment.color);
                        image.pixels[static_cast<std::size_t>(x - bounding_box.x_lower)
                                     + static_cast<std::size_t>(y - bounding_box.y_lower) * image.width] = pr.index;
                        if (dither) {
                            // Distribute error onto neighbouring points (Floyd-Steinberg, serpentine)
                            const std::array<std::array<int, 2>, 4> points = {
                                {{x + step, y}, {x - step, y + 1}, {x, y + 1}, {x + step, y + 1}}};
                            constexpr std::array<float, 4> weights = {7.0f / 16.0f, 3.0f / 16.0f, 5.0f / 16.0f,
                                                                      1.0f / 16.0f};
                            for (int i = 0; i < 4; i++) {
                                const int px = points[i][0], py = points[i][1];
                                if (px >= 0 && std::cmp_less(px, framebuffer.width) && py >= 0
                                    && std::cmp_less(py, framebuffer.height)
                                    && (!(fragment.flags & MATERIAL_NO_BLEED)
                                        || (framebuffer.fragments[px + py * framebuffer.width].flags
                                            & MATERIAL_NO_BLEED))) {
                                    framebuffer.fragments[px + py * framebuffer.width].color =
                                        vector3_add(vector3_mult(pr.error, 0.3f * weights[i]),
                                                    framebuffer.fragments[px + py * framebuffer.width].color);
                                }
                            }
                        }
                    }
                }
            }
            return image;
        }

        Image context_render_view_internal(Context& ctx, Matrix3 view, bool silhouette) {
            Matrix3 const camera = matrix_mult(ctx.projection, view);

            Rect const bounds = scene_get_bounds(ctx.rt_scene, camera);

            Framebuffer framebuffer;
            framebuffer.width = static_cast<std::uint16_t>(bounds.x_upper - bounds.x_lower + 1);
            framebuffer.height = static_cast<std::uint16_t>(bounds.y_upper - bounds.y_lower);
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
                    std::uint8_t flags = 0;
                    int region = kFragmentUnused;
                    float depth = std::numeric_limits<float>::infinity();
                    bool mask = false;
                    auto center = scene_sample_material(ctx.rt_scene, sample_point, camera_inverse);
                    float const ghost_depth = center.ghost_depth;
                    if (center) {
                        mask = center.is_mask;
                        region = mask ? kFragmentUnused : center.material->region;
                        flags = static_cast<std::uint8_t>(center.material->flags);
                        if (center.material->flags & MATERIAL_IS_VISIBLE_MASK) mask = true;
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
                                scene_sample_point(ctx.rt_scene, vector2_add(sample_point, subsample_point),
                                                   camera_inverse, transformed_lights, sub_frag);
                            } else {
                                auto sub = scene_sample_material(
                                    ctx.rt_scene, vector2_add(sample_point, subsample_point), camera_inverse);
                                sub_frag.ghost_depth = sub.ghost_depth;
                                if (sub) {
                                    sub_frag.color = vector3(0.5f, 0.5f, 0.5f);
                                    sub_frag.region = sub.is_mask ? kFragmentUnused : sub.material->region;
                                    sub_frag.flags = static_cast<std::uint8_t>(sub.material->flags);
                                    sub_frag.depth = sub.depth;
                                }
                            }
                        }

                    // Get frontmost background AA sample
                    int front_background_aa_sample = -1;
                    float min_depth = std::numeric_limits<float>::infinity();
                    for (int i = 0; i < kAaSamplesU * kAaSamplesV; i++) {
                        if (subsamples[i].depth < min_depth
                            && (subsamples[i].flags & (MATERIAL_BACKGROUND_AA | MATERIAL_BACKGROUND_AA_DARK))) {
                            front_background_aa_sample = i;
                            min_depth = subsamples[i].depth;
                        }
                    }

                    // If there exists a sample forward of the center point with background AA enabled,
                    // use that instead of the center point
                    if (front_background_aa_sample != -1 && (min_depth < ghost_depth - 4 || mask)) {
                        // Count samples that fall inside the presumed edge
                        int inside_samples = 0;
                        for (int i = 0; i < kAaSamplesU * kAaSamplesV; i++) {
                            if (!(subsamples[i].depth > min_depth + 4
                                  || (subsamples[i].region == kFragmentUnused
                                      && !(subsamples[i].flags & MATERIAL_IS_MASK))
                                  || (subsamples[i].flags & MATERIAL_IS_VISIBLE_MASK)))
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

                    if (flags & (MATERIAL_BACKGROUND_AA | MATERIAL_BACKGROUND_AA_DARK)) {
                        // Count samples that fall outside the presumed edge
                        Vector3 color = vector3(0, 0, 0);
                        float weight = 0;
                        float total_weight = 0;
                        for (int i = 0; i < kAaSamplesU * kAaSamplesV; i++) {
                            if ((!(subsamples[i].flags & MATERIAL_NO_BLEED) || (flags & MATERIAL_NO_BLEED))
                                && !((subsamples[i].ghost_depth <= depth + 4 && subsamples[i].depth > depth + 4))) {
                                if (!(subsamples[i].depth > depth + 4
                                      || (subsamples[i].region == kFragmentUnused
                                          && !(subsamples[i].flags & MATERIAL_IS_MASK))
                                      || (subsamples[i].flags & MATERIAL_IS_VISIBLE_MASK))) {
                                    color = vector3_add(color, vector3_mult(subsamples[i].color, kAaSampleWeight));
                                    weight += kAaSampleWeight;
                                }
                                total_weight += kAaSampleWeight;
                            }
                        }
                        color = vector3_mult(color, 1.0f / total_weight);
                        if (flags & MATERIAL_BACKGROUND_AA_DARK)
                            framebuffer.fragments[x + y * framebuffer.width].color =
                                vector3_mult(color, 0.5f + 0.5f * (weight / total_weight));
                        else
                            framebuffer.fragments[x + y * framebuffer.width].color = color;
                    } else {
                        Vector3 color = vector3(0, 0, 0);
                        float weight = 0.0f;
                        for (int i = 0; i < kAaSamplesU * kAaSamplesV; i++) {
                            if (subsamples[i].region != kFragmentUnused
                                && (!(subsamples[i].flags & MATERIAL_NO_BLEED) || (flags & MATERIAL_NO_BLEED))) {
                                color = vector3_add(color, vector3_mult(subsamples[i].color, kAaSampleWeight));
                                weight += kAaSampleWeight;
                            }
                        }
                        framebuffer.fragments[x + y * framebuffer.width].color = vector3_mult(color, 1.0f / weight);
                    }
                }
            };

            parallel_for(framebuffer.height, render_row);

            // Convert to indexed color
            return image_from_framebuffer(framebuffer, ctx.palette, ctx.dither);
        }
    } // namespace

    Image context_render_view(Context& ctx, Matrix3 view) {
        return context_render_view_internal(ctx, view, /*silhouette=*/false);
    }

    Image context_render_silhouette(Context& ctx, Matrix3 view) {
        return context_render_view_internal(ctx, view, /*silhouette=*/true);
    }
} // namespace RCTGen
