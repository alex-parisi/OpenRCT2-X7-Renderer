/// RayTrace.hpp

#pragma once

#include <array>
#include <bitset>
#include <cstdint>
#include <functional>
#include <type_traits>

#include <embree4/rtcore.h>

#include "Mesh.hpp"
#include "VectorMath.hpp"

namespace RCTGen {
    inline constexpr std::size_t kMaxMeshes = 128;

    using Device = RTCDevice;

    struct Vertex {
        Vector3 vertex;
        Vector3 normal;
    };

    // Mesh flag bits
    inline constexpr int MESH_MASK = 1 << 0;
    inline constexpr int MESH_GHOST = 1 << 1;

    struct Scene {
        std::array<const Mesh*, kMaxMeshes> meshes{};
        std::bitset<kMaxMeshes> mask;
        std::bitset<kMaxMeshes> ghost;
        std::uint32_t num_meshes{};
        float x_min{}, x_max{}, y_min{}, y_max{}, z_min{}, z_max{};
        RTCDevice embree_device{};
        RTCScene embree_scene{};
    };

    struct RayHit {
        std::uint32_t mesh_index{};
        std::uint32_t face_index{};
        Vector3 position{};
        Vector3 normal{};
        float ghost_distance{};
        float distance{};
        float u{};
        float v{};
    };

    Device device_init();

    void device_destroy(Device device);

    void scene_init(Scene& scene, Device device);

    void scene_finalize(Scene& scene);

    void scene_destroy(Scene& scene);

    void scene_add_model(Scene& scene,
                         const Mesh& mesh,
                         const std::function<Vertex(Vector3, Vector3, bool)>& transform_fn,
                         int flags);

    bool scene_trace_ray(Scene& scene, Vector3 origin, Vector3 direction, RayHit& hit);

    bool scene_trace_occlusion_ray(Scene& scene, Vector3 origin, Vector3 direction);

    bool scene_is_mask(Scene& scene, int index);

    bool scene_is_ghost(Scene& scene, int index);
} // namespace RCTGen
