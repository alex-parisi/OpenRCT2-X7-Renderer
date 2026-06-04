/// RayTrace.hpp

#pragma once

#include <array>
#include <bitset>
#include <cstdint>
#include <functional>

#include <embree4/rtcore.h>

#include "Mesh.hpp"
#include "VectorMath.hpp"

namespace RCTGen {
    inline constexpr std::size_t kMaxMeshes = 128;

    class DeviceHandle {
        RTCDevice device_{nullptr};
        explicit DeviceHandle(RTCDevice d) noexcept : device_(d) {}

    public:
        DeviceHandle() noexcept = default;
        ~DeviceHandle();

        [[nodiscard]] static DeviceHandle create();

        DeviceHandle(DeviceHandle&& other) noexcept;
        DeviceHandle& operator=(DeviceHandle&& other) noexcept;

        DeviceHandle(const DeviceHandle&) = delete;
        DeviceHandle& operator=(const DeviceHandle&) = delete;

        [[nodiscard]] RTCDevice get() const noexcept { return device_; }
        explicit operator bool() const noexcept { return device_ != nullptr; }
    };

    struct Vertex {
        Vector3 vertex;
        Vector3 normal;
    };

    enum class MeshFlag : int {
        None  = 0,
        Mask  = 1 << 0,
        Ghost = 1 << 1,
    };

    constexpr MeshFlag operator|(MeshFlag a, MeshFlag b) noexcept {
        return static_cast<MeshFlag>(static_cast<int>(a) | static_cast<int>(b));
    }
    constexpr MeshFlag operator&(MeshFlag a, MeshFlag b) noexcept {
        return static_cast<MeshFlag>(static_cast<int>(a) & static_cast<int>(b));
    }
    constexpr bool has_flag(MeshFlag flags, MeshFlag test) noexcept {
        return (flags & test) != MeshFlag::None;
    }

    class Scene {
    public:
        explicit Scene(RTCDevice device);
        ~Scene();

        Scene(Scene&& other) noexcept;
        Scene& operator=(Scene&& other) noexcept;

        Scene(const Scene&) = delete;
        Scene& operator=(const Scene&) = delete;

        std::array<const Mesh*, kMaxMeshes> meshes{};
        std::bitset<kMaxMeshes> mask;
        std::bitset<kMaxMeshes> ghost;
        std::uint32_t num_meshes{};
        float x_min{}, x_max{}, y_min{}, y_max{}, z_min{}, z_max{};
        RTCDevice embree_device{nullptr};
        RTCScene embree_scene{nullptr};

        void finalize();
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


    void SceneAddModel(Scene& scene,
                         const Mesh& mesh,
                         const std::function<Vertex(Vector3, Vector3)>& transform_fn,
                         MeshFlag flags);

    [[nodiscard]] bool scene_trace_ray(Scene& scene, Vector3 origin, Vector3 direction, RayHit& hit);

    [[nodiscard]] bool scene_trace_occlusion_ray(Scene& scene, Vector3 origin, Vector3 direction);

    [[nodiscard]] bool scene_is_mask(const Scene& scene, int index);

    [[nodiscard]] bool scene_is_ghost(const Scene& scene, int index);
} // namespace RCTGen
