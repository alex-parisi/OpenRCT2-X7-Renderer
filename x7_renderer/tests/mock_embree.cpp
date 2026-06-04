/// mock_embree.cpp — Mock implementations of every Embree 4 C function used by RayTrace.cpp.
///
/// The mock provides a configurable global State so tests can make individual
/// API calls return null / error codes, exercising the error-handling paths
/// that are impossible to trigger against a real Embree installation.

#include "mock_embree.h"
#include <cstring>

namespace MockEmbree {
    static State g_state;
    State& state() {
        return g_state;
    }
    void reset() {
        g_state = State{};
    }
} // namespace MockEmbree

// ---------------------------------------------------------------------------
// Fake opaque handles — the code only stores and passes these around, so a
// non-null sentinel is sufficient.
// ---------------------------------------------------------------------------
static char g_fake_device_storage;
static char g_fake_scene_storage;
static char g_fake_geometry_storage;

static RTCDevice fake_device() {
    return reinterpret_cast<RTCDevice>(&g_fake_device_storage);
}
static RTCScene fake_scene() {
    return reinterpret_cast<RTCScene>(&g_fake_scene_storage);
}
static RTCGeometry fake_geom() {
    return reinterpret_cast<RTCGeometry>(&g_fake_geometry_storage);
}

// Geometry buffer backing store — three buffers (vertices, normals, indices).
// 128 meshes * max reasonable verts — we only need enough to not segfault.
static float g_vertex_buf[1024 * 3];
static float g_normal_buf[1024 * 3];
static unsigned int g_index_buf[1024 * 3];
static int g_buffer_call = 0;

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------
extern "C" {

RTCDevice rtcNewDevice(const char*) {
    auto& s = MockEmbree::state();
    if (s.device_init_fails) return nullptr;
    return fake_device();
}

enum RTCError rtcGetDeviceError(RTCDevice) {
    return RTC_ERROR_UNKNOWN;
}

void rtcSetDeviceErrorFunction(RTCDevice, RTCErrorFunction fn, void* ptr) {
    auto& s = MockEmbree::state();
    s.error_callback = fn;
    s.error_callback_user_ptr = ptr;
}

void rtcReleaseDevice(RTCDevice) {
}

// ---------------------------------------------------------------------------
// Scene
// ---------------------------------------------------------------------------

RTCScene rtcNewScene(RTCDevice) {
    return fake_scene();
}
void rtcCommitScene(RTCScene) {
}
void rtcReleaseScene(RTCScene) {
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

RTCGeometry rtcNewGeometry(RTCDevice, enum RTCGeometryType) {
    auto& s = MockEmbree::state();
    if (s.geometry_alloc_fails) return nullptr;
    return fake_geom();
}

void rtcSetGeometryVertexAttributeCount(RTCGeometry, unsigned int) {
}

void* rtcSetNewGeometryBuffer(RTCGeometry, enum RTCBufferType type, unsigned int, enum RTCFormat, size_t, size_t) {
    auto& s = MockEmbree::state();
    if (s.buffer_alloc_fails) return nullptr;
    g_buffer_call++;
    if (type == RTC_BUFFER_TYPE_VERTEX) {
        if (s.vertex_buffer_fails) return nullptr;
        return g_vertex_buf;
    }
    if (type == RTC_BUFFER_TYPE_VERTEX_ATTRIBUTE) {
        if (s.normal_buffer_fails) return nullptr;
        return g_normal_buf;
    }
    if (type == RTC_BUFFER_TYPE_INDEX) {
        if (s.index_buffer_fails) return nullptr;
        return g_index_buf;
    }
    return g_vertex_buf;
}

void rtcReleaseGeometry(RTCGeometry) {
}
void rtcCommitGeometry(RTCGeometry) {
}
unsigned int rtcAttachGeometry(RTCScene, RTCGeometry) {
    return 0;
}
void rtcSetGeometryOccludedFilterFunction(RTCGeometry, RTCFilterFunctionN) {
}

// ---------------------------------------------------------------------------
// Ray tracing — not needed for error-path tests, but must exist to link.
// ---------------------------------------------------------------------------

void rtcIntersect1(RTCScene, struct RTCRayHit* rayhit, struct RTCIntersectArguments*) {
    rayhit->hit.geomID = RTC_INVALID_GEOMETRY_ID;
}

void rtcOccluded1(RTCScene, struct RTCRay*, struct RTCOccludedArguments*) {
}

RTCGeometry rtcGetGeometry(RTCScene, unsigned int) {
    return fake_geom();
}

void rtcInterpolate(const struct RTCInterpolateArguments* args) {
    if (args->P) std::memset(args->P, 0, args->valueCount * sizeof(float));
}

} // extern "C"
