/// test_ray_trace.cpp

#include "Mesh.hpp"
#include "RayTrace.hpp"
#include "VectorMath.hpp"
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using namespace RCTGen;

namespace {
    struct TestMeshData {
        std::vector<Vector3> vertices;
        std::vector<Vector3> normals;
        std::vector<Vector2> uvs;
        std::vector<Face> faces;
        std::vector<Material> materials;
        Mesh mesh;

        static TestMeshData make_triangle() {
            TestMeshData d;
            d.vertices = {{-1, -1, 0}, {1, -1, 0}, {0, 1, 0}};
            d.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
            d.uvs = {{0, 0}, {1, 0}, {0.5f, 1}};
            d.faces = {{0, {0, 1, 2}}};
            d.materials = {Material{}};
            d.mesh = {d.vertices, d.normals, d.uvs, d.faces, d.materials};
            return d;
        }

        static TestMeshData make_quad() {
            TestMeshData d;
            d.vertices = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}};
            d.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
            d.uvs = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
            d.faces = {{0, {0, 1, 2}}, {0, {0, 2, 3}}};
            d.materials = {Material{}};
            d.mesh = {d.vertices, d.normals, d.uvs, d.faces, d.materials};
            return d;
        }
    };

    Vertex identity_transform(Vector3 v, Vector3 n) {
        return {v, n};
    }
} // namespace

class RayTraceTest : public ::testing::Test {
protected:
    DeviceHandle device;

    void SetUp() override {
        device = DeviceHandle::create();
    }
    void TearDown() override {
        device = {};
    }
};

TEST_F(RayTraceTest, DeviceInitReturnsNonNull) {
    EXPECT_NE(device.get(), nullptr);
}

TEST_F(RayTraceTest, DeviceHandleBoolOperator) {
    EXPECT_TRUE(static_cast<bool>(device));
    DeviceHandle empty;
    EXPECT_FALSE(static_cast<bool>(empty));
}

TEST_F(RayTraceTest, DeviceHandleMoveConstructor) {
    DeviceHandle moved(std::move(device));
    EXPECT_TRUE(static_cast<bool>(moved));
    EXPECT_EQ(device.get(), nullptr);
}

TEST_F(RayTraceTest, SceneLifecycle) {
    Scene scene(device.get());
    scene.finalize();
}

TEST_F(RayTraceTest, SceneAddModelSetsFlags) {
    auto data = TestMeshData::make_triangle();
    Scene scene(device.get());
    SceneAddModel(scene, data.mesh, identity_transform, MeshFlag::Mask | MeshFlag::Ghost);
    EXPECT_TRUE(scene_is_mask(scene, 0));
    EXPECT_TRUE(scene_is_ghost(scene, 0));
    EXPECT_EQ(scene.num_meshes, 1u);
    scene.finalize();
}

TEST_F(RayTraceTest, SceneAddModelNoFlags) {
    auto data = TestMeshData::make_triangle();
    Scene scene(device.get());
    SceneAddModel(scene, data.mesh, identity_transform, MeshFlag::None);
    EXPECT_FALSE(scene_is_mask(scene, 0));
    EXPECT_FALSE(scene_is_ghost(scene, 0));
    scene.finalize();
}

TEST_F(RayTraceTest, TraceRayHitsTriangle) {
    auto data = TestMeshData::make_triangle();
    Scene scene(device.get());
    SceneAddModel(scene, data.mesh, identity_transform, MeshFlag::None);
    scene.finalize();

    RayHit hit{};
    bool result = scene_trace_ray(scene, vector3(0, 0, -5), vector3(0, 0, 1), hit);
    EXPECT_TRUE(result);
    EXPECT_EQ(hit.mesh_index, 0u);
    EXPECT_EQ(hit.face_index, 0u);
    EXPECT_NEAR(hit.normal.z, 1.0f, 0.01f);
    EXPECT_GT(hit.distance, 0.0f);
}

TEST_F(RayTraceTest, TraceRayMisses) {
    auto data = TestMeshData::make_triangle();
    Scene scene(device.get());
    SceneAddModel(scene, data.mesh, identity_transform, MeshFlag::None);
    scene.finalize();

    RayHit hit{};
    bool result = scene_trace_ray(scene, vector3(10, 10, -5), vector3(0, 0, 1), hit);
    EXPECT_FALSE(result);
}

TEST_F(RayTraceTest, TraceRaySkipsGhostMesh) {
    auto ghost_data = TestMeshData::make_quad();
    auto solid_data = TestMeshData::make_quad();

    // Place ghost at z=0, solid at z=2 (behind ghost from ray's perspective)
    auto ghost_xform = [](Vector3 v, Vector3 n) -> Vertex {
        return {v, n};
    };
    auto solid_xform = [](Vector3 v, Vector3 n) -> Vertex {
        return {{v.x, v.y, v.z + 2.0f}, n};
    };

    Scene scene(device.get());
    SceneAddModel(scene, ghost_data.mesh, ghost_xform, MeshFlag::Ghost);
    SceneAddModel(scene, solid_data.mesh, solid_xform, MeshFlag::None);
    scene.finalize();

    RayHit hit{};
    bool result = scene_trace_ray(scene, vector3(0, 0, -5), vector3(0, 0, 1), hit);
    EXPECT_TRUE(result);
    EXPECT_EQ(hit.mesh_index, 1u);
}

TEST_F(RayTraceTest, OcclusionRayHitsFrontFace) {
    auto data = TestMeshData::make_triangle();
    Scene scene(device.get());
    SceneAddModel(scene, data.mesh, identity_transform, MeshFlag::None);
    scene.finalize();

    // Ray from +z toward -z hits the front face (Ng=(0,0,1), dir=(0,0,-1), dot < 0 → not culled)
    bool occluded = scene_trace_occlusion_ray(scene, vector3(0, 0, 5), vector3(0, 0, -1));
    EXPECT_TRUE(occluded);
}

TEST_F(RayTraceTest, OcclusionRayMisses) {
    auto data = TestMeshData::make_triangle();
    Scene scene(device.get());
    SceneAddModel(scene, data.mesh, identity_transform, MeshFlag::None);
    scene.finalize();

    bool occluded = scene_trace_occlusion_ray(scene, vector3(10, 10, -5), vector3(0, 0, 1));
    EXPECT_FALSE(occluded);
}

TEST_F(RayTraceTest, OcclusionRayBackFaceIsCulled) {
    auto data = TestMeshData::make_triangle();
    Scene scene(device.get());
    SceneAddModel(scene, data.mesh, identity_transform, MeshFlag::None);
    scene.finalize();

    // Ray from -z toward +z hits the back face (Ng=(0,0,1), dir=(0,0,1), dot > 0 → culled)
    bool occluded = scene_trace_occlusion_ray(scene, vector3(0, 0, -5), vector3(0, 0, 1));
    EXPECT_FALSE(occluded);
}

TEST_F(RayTraceTest, SceneAddModelWithFlatShading) {
    auto data = TestMeshData::make_triangle();
    data.materials[0].flags = MaterialFlag::IsFlatShaded;
    data.mesh = {data.vertices, data.normals, data.uvs, data.faces, data.materials};

    Scene scene(device.get());
    SceneAddModel(scene, data.mesh, identity_transform, MeshFlag::None);
    scene.finalize();

    RayHit hit{};
    bool result = scene_trace_ray(scene, vector3(0, 0, -5), vector3(0, 0, 1), hit);
    EXPECT_TRUE(result);
}

TEST_F(RayTraceTest, SceneAddModelUpdatesAABB) {
    auto data = TestMeshData::make_triangle();
    Scene scene(device.get());
    SceneAddModel(scene, data.mesh, identity_transform, MeshFlag::None);
    EXPECT_LE(scene.x_min, -1.0f);
    EXPECT_GE(scene.x_max, 1.0f);
    EXPECT_LE(scene.y_min, -1.0f);
    EXPECT_GE(scene.y_max, 1.0f);
    scene.finalize();
}

TEST_F(RayTraceTest, SceneAddModelExceedsMeshLimit) {
    auto data = TestMeshData::make_triangle();
    Scene scene(device.get());
    for (std::size_t i = 0; i < kMaxMeshes; ++i)
        SceneAddModel(scene, data.mesh, identity_transform, MeshFlag::None);
    EXPECT_THROW(SceneAddModel(scene, data.mesh, identity_transform, MeshFlag::None), std::runtime_error);
    scene.finalize();
}

TEST_F(RayTraceTest, SceneMoveConstructor) {
    auto data = TestMeshData::make_triangle();
    Scene scene(device.get());
    SceneAddModel(scene, data.mesh, identity_transform, MeshFlag::None);
    scene.finalize();

    Scene moved(std::move(scene));
    RayHit hit{};
    bool result = scene_trace_ray(moved, vector3(0, 0, -5), vector3(0, 0, 1), hit);
    EXPECT_TRUE(result);
    EXPECT_EQ(scene.embree_scene, nullptr);
}

TEST_F(RayTraceTest, SceneMoveAssignment) {
    auto data = TestMeshData::make_triangle();
    Scene scene(device.get());
    SceneAddModel(scene, data.mesh, identity_transform, MeshFlag::None);
    scene.finalize();

    Scene other(device.get());
    other = std::move(scene);
    RayHit hit{};
    bool result = scene_trace_ray(other, vector3(0, 0, -5), vector3(0, 0, 1), hit);
    EXPECT_TRUE(result);
    EXPECT_EQ(scene.embree_scene, nullptr);
}
