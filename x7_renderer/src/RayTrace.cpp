/// RayTrace.cpp

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>

#include "RayTrace.hpp"
#include <embree4/rtcore.h>

namespace RCTGen {
    namespace {
        void rt_error(void* /*user_ptr*/, enum RTCError error, const char* str) {
            std::fprintf(stderr, "error %d: %s\n", static_cast<int>(error), str);
            std::exit(1);
        }

        void occlusionFilter(const struct RTCFilterFunctionNArguments* args) {
            // Check that packet size is 1 (guaranteed by Embree for scalar calls)
            [[maybe_unused]] const unsigned int N = args->N;
            assert(N == 1);

            // RTCRayN/RTCHitN are opaque incomplete types in Embree's API;
            // reinterpret_cast is the correct tool here.
            auto const* ray = reinterpret_cast<struct RTCRay*>(args->ray);
            auto const* hit = reinterpret_cast<struct RTCHit*>(args->hit);

            if (hit->Ng_x * ray->dir_x + hit->Ng_y * ray->dir_y + hit->Ng_z * ray->dir_z > 0) args->valid[0] = 0;
        }
    } // namespace

    Device device_init() {
        Device device = rtcNewDevice(nullptr);
        if (!device) {
            std::fprintf(stderr, "error %d: cannot create device\n", static_cast<int>(rtcGetDeviceError(nullptr)));
            std::exit(1);
        }
        rtcSetDeviceErrorFunction(device, rt_error, nullptr);
        return device;
    }

    void device_destroy(Device device) { rtcReleaseDevice(device); }

    bool scene_is_mask(Scene& scene, int index) { return scene.mask.test(static_cast<std::size_t>(index)); }

    bool scene_is_ghost(Scene& scene, int index) { return scene.ghost.test(static_cast<std::size_t>(index)); }

    void scene_init(Scene& scene, Device device) {
        scene.num_meshes = 0;
        scene.mask.reset();
        scene.ghost.reset();
        scene.embree_device = device;
        scene.embree_scene = rtcNewScene(device);
        scene.x_max = -std::numeric_limits<float>::infinity();
        scene.y_max = -std::numeric_limits<float>::infinity();
        scene.z_max = -std::numeric_limits<float>::infinity();
        scene.x_min = std::numeric_limits<float>::infinity();
        scene.y_min = std::numeric_limits<float>::infinity();
        scene.z_min = std::numeric_limits<float>::infinity();
    }

    void scene_finalize(Scene& scene) { rtcCommitScene(scene.embree_scene); }

    void scene_destroy(Scene& scene) { rtcReleaseScene(scene.embree_scene); }

    void scene_add_model(Scene& scene,
                         const Mesh& mesh,
                         const std::function<Vertex(Vector3, Vector3, bool)>& transform_fn,
                         int flags) {
        // Add mesh to list of meshes
        assert(scene.num_meshes < kMaxMeshes);
        scene.meshes[scene.num_meshes] = &mesh;
        if (flags & MESH_MASK) scene.mask.set(scene.num_meshes);
        if (flags & MESH_GHOST) scene.ghost.set(scene.num_meshes);
        scene.num_meshes++;

        // Create Embree geometry
        RTCGeometry geom = rtcNewGeometry(scene.embree_device, RTC_GEOMETRY_TYPE_TRIANGLE);
        if (geom == nullptr) {
            std::fprintf(stderr, "Failed allocating geometry\n");
            return;
        }

        rtcSetGeometryVertexAttributeCount(geom, 1);
        auto* vertices = static_cast<float*>(rtcSetNewGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
                                                                     3 * sizeof(float), mesh.vertices.size()));
        auto* normals = static_cast<float*>(rtcSetNewGeometryBuffer(
            geom, RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float), mesh.vertices.size()));
        auto* indices = static_cast<unsigned int*>(rtcSetNewGeometryBuffer(
            geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof(unsigned int), mesh.faces.size()));
        if (!(vertices && indices && normals)) {
            std::fprintf(stderr, "Failed allocating geometry buffer\n");
            rtcReleaseGeometry(geom);
            return;
        }

        // Flat-shading is a per-mesh property: pre-compute once instead of re-scanning
        // all faces for every vertex (was O(V*F), now O(F+V)).
        const bool flat_shaded = std::ranges::any_of(mesh.faces, [&mesh](const Face& f) {
            return static_cast<bool>(mesh.materials[f.material].flags & MATERIAL_IS_FLAT_SHADED);
        });

        for (std::size_t i = 0; i < mesh.vertices.size(); i++) {
            Vertex const transformed_vertex = transform_fn(mesh.vertices[i], mesh.normals[i], flat_shaded);
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
        rtcSetGeometryOccludedFilterFunction(geom, occlusionFilter);
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

        // If we hit ghost mesh, keep tracing
        while ((rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID)
               && scene_is_ghost(scene, static_cast<int>(rayhit.hit.geomID))) {
            rayhit.ray.tnear = rayhit.ray.tfar + 0.0001f;
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
