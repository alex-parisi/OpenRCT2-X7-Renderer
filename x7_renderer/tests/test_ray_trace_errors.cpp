/// test_ray_trace_errors.cpp — Tests for Embree error-handling paths in RayTrace.cpp.
///
/// These tests run against mock_embree.cpp, allowing us to make Embree API
/// calls return null/error to exercise the otherwise-unreachable error paths.

#include "Mesh.hpp"
#include "RayTrace.hpp"
#include "VectorMath.hpp"
#include "mock_embree.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using namespace RCTGen;

namespace {
    Vertex identity_transform(Vector3 v, Vector3 n) {
        return {v, n};
    }

    struct TestMeshData {
        std::vector<Vector3> vertices{{-1, -1, 0}, {1, -1, 0}, {0, 1, 0}};
        std::vector<Vector3> normals{{0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
        std::vector<Vector2> uvs{{0, 0}, {1, 0}, {0.5f, 1}};
        std::vector<Face> faces{{0, {0, 1, 2}}};
        std::vector<Material> materials{Material{}};
        Mesh mesh{vertices, normals, uvs, faces, materials};
    };
} // namespace

class RayTraceErrorTest : public ::testing::Test {
protected:
    void SetUp() override {
        MockEmbree::reset();
    }
    void TearDown() override {
        MockEmbree::reset();
    }
};

TEST_F(RayTraceErrorTest, DeviceInitThrowsWhenEmbreeReturnsNull) {
    MockEmbree::state().device_init_fails = true;
    EXPECT_THROW(DeviceHandle::create(), std::runtime_error);
}

TEST_F(RayTraceErrorTest, DeviceInitErrorMessageContainsErrorCode) {
    MockEmbree::state().device_init_fails = true;
    try {
        DeviceHandle::create();
        FAIL() << "Expected runtime_error";
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("cannot create device"), std::string::npos);
    }
}

TEST_F(RayTraceErrorTest, SceneAddModelThrowsWhenGeometryAllocFails) {
    TestMeshData data;
    auto device = DeviceHandle::create();
    Scene scene(device.get());
    MockEmbree::state().geometry_alloc_fails = true;
    EXPECT_THROW(SceneAddModel(scene, data.mesh, identity_transform, MeshFlag::None), std::runtime_error);
}

TEST_F(RayTraceErrorTest, SceneAddModelThrowsWhenBufferAllocFails) {
    TestMeshData data;
    auto device = DeviceHandle::create();
    Scene scene(device.get());
    MockEmbree::state().buffer_alloc_fails = true;
    EXPECT_THROW(SceneAddModel(scene, data.mesh, identity_transform, MeshFlag::None), std::runtime_error);
}

TEST_F(RayTraceErrorTest, RtErrorCallbackThrowsRuntimeError) {
    auto device = DeviceHandle::create();
    auto& s = MockEmbree::state();
    ASSERT_NE(s.error_callback, nullptr);
    EXPECT_THROW(s.error_callback(s.error_callback_user_ptr, RTC_ERROR_UNKNOWN, "test error"), std::runtime_error);
}

TEST_F(RayTraceErrorTest, RtErrorMessageContainsErrorCodeAndString) {
    auto device = DeviceHandle::create();
    auto& s = MockEmbree::state();
    try {
        s.error_callback(s.error_callback_user_ptr, RTC_ERROR_UNKNOWN, "some failure");
        FAIL() << "Expected runtime_error";
    } catch (const std::runtime_error& e) {
        std::string msg = e.what();
        EXPECT_NE(msg.find("Embree error"), std::string::npos);
        EXPECT_NE(msg.find("some failure"), std::string::npos);
    }
}

TEST_F(RayTraceErrorTest, BufferFailOnNormalsOnly) {
    TestMeshData data;
    auto device = DeviceHandle::create();
    Scene scene(device.get());
    MockEmbree::state().normal_buffer_fails = true;
    EXPECT_THROW(SceneAddModel(scene, data.mesh, identity_transform, MeshFlag::None), std::runtime_error);
}

TEST_F(RayTraceErrorTest, BufferFailOnIndicesOnly) {
    TestMeshData data;
    auto device = DeviceHandle::create();
    Scene scene(device.get());
    MockEmbree::state().index_buffer_fails = true;
    EXPECT_THROW(SceneAddModel(scene, data.mesh, identity_transform, MeshFlag::None), std::runtime_error);
}
