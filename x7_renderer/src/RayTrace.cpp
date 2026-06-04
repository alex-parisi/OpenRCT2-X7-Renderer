/// RayTrace.cpp

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "RayTrace.hpp"
#include <embree4/rtcore.h>

namespace RCTGen {
    // Minimum tnear offset when re-tracing past a ghost mesh hit.
    // Chosen to be larger than any floating-point rounding error in Embree's
    // BVH traversal at the scene scales used (units-per-tile ~ 3-32).
    constexpr float kGhostRayEpsilon = 1e-4f;

    namespace {
        [[noreturn]] void RtError(void* /*user_ptr*/, enum RTCError error, const char* str) {
            throw std::runtime_error("Embree error " + std::to_string(static_cast<int>(error)) + ": " + str);
        }

        void OcclusionFilter(const struct RTCFilterFunctionNArguments* args) {
            // Check that packet size is 1 (guaranteed by Embree for scalar calls)
            [[maybe_unused]] const unsigned int n = args->N;
            assert(n == 1);

            // RTCRayN/RTCHitN are opaque incomplete types in Embree's API;
            // reinterpret_cast is the correct tool here.
            auto const* ray = reinterpret_cast<struct RTCRay*>(args->ray);
            auto const* hit = reinterpret_cast<struct RTCHit*>(args->hit);

            if (hit->Ng_x * ray->dir_x + hit->Ng_y * ray->dir_y + hit->Ng_z * ray->dir_z > 0) args->valid[0] = 0;
        }
    } // namespace

    DeviceHandle DeviceHandle::create() {
        RTCDevice d = rtcNewDevice(nullptr);
        if (!d) {
            throw std::runtime_error("Embree: cannot create device (error "
                                     + std::to_string(static_cast<int>(rtcGetDeviceError(nullptr))) + ")");
        }
        rtcSetDeviceErrorFunction(d, RtError, nullptr);
        return DeviceHandle(d);
    }

    DeviceHandle::~DeviceHandle() {
        if (device_) rtcReleaseDevice(device_);
    }

    DeviceHandle::DeviceHandle(DeviceHandle&& other) noexcept : device_(std::exchange(other.device_, nullptr)) {
    }

    DeviceHandle& DeviceHandle::operator=(DeviceHandle&& other) noexcept {
        if (this != &other) {
            if (device_) rtcReleaseDevice(device_);
            device_ = std::exchange(other.device_, nullptr);
        }
        return *this;
    }

    bool scene_is_mask(const Scene& scene, int index) {
        return scene.mask.test(static_cast<std::size_t>(index));
    }

    bool scene_is_ghost(const Scene& scene, int index) {
        return scene.ghost.test(static_cast<std::size_t>(index));
    }

    Scene::Scene(RTCDevice device)
        : x_min(std::numeric_limits<float>::infinity()), x_max(-std::numeric_limits<float>::infinity()),
          y_min(std::numeric_limits<float>::infinity()), y_max(-std::numeric_limits<float>::infinity()),
          z_min(std::numeric_limits<float>::infinity()), z_max(-std::numeric_limits<float>::infinity()),
          embree_device(device), embree_scene(rtcNewScene(device)) {
    }

    Scene::~Scene() {
        if (embree_scene) rtcReleaseScene(embree_scene);
    }

    Scene::Scene(Scene&& other) noexcept
        : meshes(other.meshes), mask(other.mask), ghost(other.ghost), num_meshes(other.num_meshes), x_min(other.x_min),
          x_max(other.x_max), y_min(other.y_min), y_max(other.y_max), z_min(other.z_min), z_max(other.z_max),
          embree_device(other.embree_device), embree_scene(std::exchange(other.embree_scene, nullptr)) {
    }

    Scene& Scene::operator=(Scene&& other) noexcept {
        if (this != &other) {
            if (embree_scene) rtcReleaseScene(embree_scene);
            meshes = other.meshes;
            mask = other.mask;
            ghost = other.ghost;
            num_meshes = other.num_meshes;
            x_min = other.x_min;
            x_max = other.x_max;
            y_min = other.y_min;
            y_max = other.y_max;
            z_min = other.z_min;
            z_max = other.z_max;
            embree_device = other.embree_device;
            embree_scene = std::exchange(other.embree_scene, nullptr);
        }
        return *this;
    }

    void Scene::finalize() {
        rtcCommitScene(embree_scene);
    }

    // NOLINTNEXTLINE(misc-use-internal-linkage)
    void SceneAddModel(Scene& scene,
                       const Mesh& mesh,
                       const std::function<Vertex(Vector3, Vector3)>& transform_fn,
                       MeshFlag flags) {
        if (scene.num_meshes >= kMaxMeshes)
            throw std::runtime_error("scene mesh limit exceeded (max " + std::to_string(kMaxMeshes) + ")");
        scene.meshes[scene.num_meshes] = &mesh;
        if (has_flag(flags, MeshFlag::Mask)) scene.mask.set(scene.num_meshes);
        if (has_flag(flags, MeshFlag::Ghost)) scene.ghost.set(scene.num_meshes);
        scene.num_meshes++;

        // Create Embree geometry
        RTCGeometry geom = rtcNewGeometry(scene.embree_device, RTC_GEOMETRY_TYPE_TRIANGLE);
        if (geom == nullptr) throw std::runtime_error("Embree: failed to allocate geometry");

        rtcSetGeometryVertexAttributeCount(geom, 1);
        auto* vertices = static_cast<float*>(rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
                                                                     3 * sizeof(float), mesh.vertices.size()));
        auto* normals = static_cast<float*>(rtcSetNewGeometryBuffer(
            geom, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float), mesh.vertices.size()));
        auto* indices = static_cast<unsigned int*>(rtcSetNewGeometryBuffer(
            geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(unsigned int), mesh.faces.size()));
        if (!(vertices && indices && normals)) {
            rtcReleaseGeometry(geom);
            throw std::runtime_error("Embree: failed to allocate geometry buffer");
        }

        for (std::size_t i = 0; i < mesh.vertices.size(); i++) {
            Vertex const transformed_vertex = transform_fn(mesh.vertices[i], mesh.normals[i]);
            vertices[3 * i + 0] = transformed_vertex.vertex.x;
            vertices[3 * i + 1] = transformed_vertex.vertex.y;
            vertices[3 * i + 2] = transformed_vertex.vertex.z;
            normals[3 * i + 0] = transformed_vertex.normal.x;
            normals[3 * i + 1] = transformed_vertex.normal.y;
            normals[3 * i + 2] = transformed_vertex.normal.z;
            scene.x_max = std::max(scene.x_max, transformed_vertex.vertex.x);
            scene.y_max = std::max(scene.y_max, transformed_vertex.vertex.y);
            scene.z_max = std::max(scene.z_max, transformed_vertex.vertex.z);
            scene.x_min = std::min(scene.x_min, transformed_vertex.vertex.x);
            scene.y_min = std::min(scene.y_min, transformed_vertex.vertex.y);
            scene.z_min = std::min(scene.z_min, transformed_vertex.vertex.z);
        }

        for (std::size_t i = 0; i < mesh.faces.size(); i++) {
            indices[3 * i + 0] = static_cast<unsigned int>(mesh.faces[i].indices[0]);
            indices[3 * i + 1] = static_cast<unsigned int>(mesh.faces[i].indices[1]);
            indices[3 * i + 2] = static_cast<unsigned int>(mesh.faces[i].indices[2]);
        }
        rtcSetGeometryOccludedFilterFunction(geom, OcclusionFilter);
        rtcCommitGeometry(geom);
        // Add geometry to scene
        rtcAttachGeometry(scene.embree_scene, geom);
        rtcReleaseGeometry(geom);
    }

    bool scene_trace_ray(Scene& scene, Vector3 origin, Vector3 direction, RayHit& hit) {
        struct RTCRayHit rayhit;

        rayhit.ray.org_x = origin.x;
        rayhit.ray.org_y = origin.y;
        rayhit.ray.org_z = origin.z;
        rayhit.ray.dir_x = direction.x;
        rayhit.ray.dir_y = direction.y;
        rayhit.ray.dir_z = direction.z;
        rayhit.ray.tnear = 0;
        rayhit.ray.tfar = std::numeric_limits<float>::infinity();
        rayhit.ray.mask = -1;
        rayhit.ray.flags = 0;
        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

        rtcIntersect1(scene.embree_scene, &rayhit);

        hit.ghost_distance = rayhit.ray.tfar;

        // If we hit ghost mesh, keep tracing (bounded by max mesh count).
        int ghost_hops = 0;
        while ((rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID)
               && scene_is_ghost(scene, static_cast<int>(rayhit.hit.geomID))
               && ghost_hops++ < static_cast<int>(kMaxMeshes)) {
            rayhit.ray.tnear = rayhit.ray.tfar + kGhostRayEpsilon;
            rayhit.ray.tfar = std::numeric_limits<float>::infinity();
            rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
            rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
            rtcIntersect1(scene.embree_scene, &rayhit);
        }

        if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
            hit.mesh_index = rayhit.hit.geomID;
            hit.face_index = rayhit.hit.primID;
            hit.u = rayhit.hit.u;
            hit.v = rayhit.hit.v;

            // Interpolate normal
            float position_components[3];
            float normal_components[3];
            rtcInterpolate0(rtcGetGeometry(scene.embree_scene, rayhit.hit.geomID), rayhit.hit.primID, rayhit.hit.u,
                            rayhit.hit.v, RTC_BUFFER_TYPE_VERTEX, 0, position_components, 3);
            rtcInterpolate0(rtcGetGeometry(scene.embree_scene, rayhit.hit.geomID), rayhit.hit.primID, rayhit.hit.u,
                            rayhit.hit.v, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0, normal_components, 3);
            hit.position = vector3(position_components[0], position_components[1], position_components[2]);
            hit.normal = vector3_normalize(vector3(normal_components[0], normal_components[1], normal_components[2]));
            hit.distance = rayhit.ray.tfar;
            return true;
        }
        return false;
    }

    bool scene_trace_occlusion_ray(Scene& scene, Vector3 origin, Vector3 direction) {
        struct RTCRay ray;
        ray.org_x = origin.x;
        ray.org_y = origin.y;
        ray.org_z = origin.z;
        ray.dir_x = direction.x;
        ray.dir_y = direction.y;
        ray.dir_z = direction.z;
        ray.tnear = 1e-5f;
        ray.tfar = std::numeric_limits<float>::infinity();
        ray.mask = -1;
        ray.flags = 0;

        rtcOccluded1(scene.embree_scene, &ray);

        return ray.tfar <= 0.0f;
    }
} // namespace RCTGen
