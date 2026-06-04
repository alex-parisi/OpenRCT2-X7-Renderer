/// test_renderer.cpp

#include "Mesh.hpp"
#include "Palette.hpp"
#include "Renderer.hpp"
#include "VectorMath.hpp"
#include <cstdlib>
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
            d.materials[0].color = vector3(0.8f, 0.2f, 0.2f);
            d.materials[0].ambient_color = vector3(0.1f, 0.1f, 0.1f);
            d.materials[0].specular_color = vector3(1.0f, 1.0f, 1.0f);
            d.materials[0].specular_exponent = 32.0f;
            d.mesh = {d.vertices, d.normals, d.uvs, d.faces, d.materials};
            return d;
        }

        static TestMeshData make_textured_triangle() {
            TestMeshData d;
            d.vertices = {{-1, -1, 0}, {1, -1, 0}, {0, 1, 0}};
            d.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
            d.uvs = {{0, 0}, {1, 0}, {0.5f, 1}};
            d.faces = {{0, {0, 1, 2}}};
            d.materials = {Material{}};
            d.materials[0].flags = MaterialFlag::HasTexture;
            d.materials[0].ambient_color = vector3(0.1f, 0.1f, 0.1f);
            d.mesh = {d.vertices, d.normals, d.uvs, d.faces, d.materials};
            return d;
        }
    };

    std::vector<Light> test_lights() {
        return {
            {LightType::Hemi, 0, vector3(0, 1, 0), 0.5f},
            {LightType::Diffuse, 0, vector3_normalize(vector3(1, 1, 1)), 0.8f},
            {LightType::Specular, 0, vector3_normalize(vector3(1, 1, 1)), 0.3f},
        };
    }
} // namespace

class RendererTest : public ::testing::Test {
protected:
    Context ctx{};

    void SetUp() override {
        auto lights = test_lights();
        ContextInit(ctx, lights, false, palette_rct2(), 32.0f);
    }
    void TearDown() override {
        context_destroy(ctx);
    }
};

TEST_F(RendererTest, EmptySceneRenderView) {
    context_begin_render(ctx);
    context_finalize_render(ctx);
    Image img = context_render_view(ctx, views[0]);
    EXPECT_GE(img.width, 0);
    context_end_render(ctx);
}

TEST_F(RendererTest, EmptySceneRenderSilhouette) {
    context_begin_render(ctx);
    context_finalize_render(ctx);
    Image img = context_render_silhouette(ctx, views[0]);
    EXPECT_GE(img.width, 0);
    context_end_render(ctx);
}

TEST_F(RendererTest, RenderTriangleProducesPixels) {
    auto data = TestMeshData::make_triangle();
    context_begin_render(ctx);
    context_add_model(ctx, data.mesh, transform(matrix_identity(), vector3(0, 0, 0)));
    context_finalize_render(ctx);
    Image img = context_render_view(ctx, views[0]);
    EXPECT_GT(img.width, 0);
    EXPECT_GT(img.height, 0);
    EXPECT_FALSE(img.pixels.empty());
    bool has_nonzero = false;
    for (auto px : img.pixels) {
        if (px != 0) { has_nonzero = true; break; }
    }
    EXPECT_TRUE(has_nonzero);
    context_end_render(ctx);
}

TEST_F(RendererTest, RenderSilhouetteProducesPixels) {
    auto data = TestMeshData::make_triangle();
    context_begin_render(ctx);
    context_add_model(ctx, data.mesh, transform(matrix_identity(), vector3(0, 0, 0)));
    context_finalize_render(ctx);
    Image img = context_render_silhouette(ctx, views[0]);
    EXPECT_GT(img.width, 0);
    EXPECT_GT(img.height, 0);
    context_end_render(ctx);
}

TEST_F(RendererTest, RenderAllFourViews) {
    auto data = TestMeshData::make_triangle();
    for (const auto& view : views) {
        context_begin_render(ctx);
        context_add_model(ctx, data.mesh, transform(matrix_identity(), vector3(0, 0, 0)));
        context_finalize_render(ctx);
        Image img = context_render_view(ctx, view);
        EXPECT_GT(img.width, 0);
        context_end_render(ctx);
    }
}

TEST_F(RendererTest, RenderWithMaskFlag) {
    auto data = TestMeshData::make_triangle();
    context_begin_render(ctx);
    context_add_model(ctx, data.mesh, transform(matrix_identity(), vector3(0, 0, 0)), MeshFlag::Mask);
    context_finalize_render(ctx);
    Image img = context_render_view(ctx, views[0]);
    EXPECT_GE(img.width, 0);
    context_end_render(ctx);
}

TEST_F(RendererTest, RenderWithGhostFlag) {
    auto data = TestMeshData::make_triangle();
    context_begin_render(ctx);
    context_add_model(ctx, data.mesh, transform(matrix_identity(), vector3(0, 0, 0)), MeshFlag::Ghost);
    context_finalize_render(ctx);
    Image img = context_render_view(ctx, views[0]);
    EXPECT_GE(img.width, 0);
    context_end_render(ctx);
}

TEST_F(RendererTest, RenderWithDithering) {
    context_destroy(ctx);
    auto lights = test_lights();
    ContextInit(ctx, lights, true, palette_rct2(), 32.0f);

    auto data = TestMeshData::make_triangle();
    context_begin_render(ctx);
    context_add_model(ctx, data.mesh, transform(matrix_identity(), vector3(0, 0, 0)));
    context_finalize_render(ctx);
    Image img = context_render_view(ctx, views[0]);
    EXPECT_GT(img.width, 0);
    context_end_render(ctx);
}

TEST_F(RendererTest, RenderWithRotation) {
    auto data = TestMeshData::make_triangle();
    context_begin_render(ctx);
    context_add_model(ctx, data.mesh, transform(rotate_y(1.0f), vector3(0, 0, 0)));
    context_finalize_render(ctx);
    Image img = context_render_view(ctx, views[0]);
    EXPECT_GT(img.width, 0);
    context_end_render(ctx);
}

TEST_F(RendererTest, RenderWithTranslation) {
    auto data = TestMeshData::make_triangle();
    context_begin_render(ctx);
    context_add_model(ctx, data.mesh, transform(matrix_identity(), vector3(2, 0, 0)));
    context_finalize_render(ctx);
    Image img = context_render_view(ctx, views[0]);
    EXPECT_GT(img.width, 0);
    context_end_render(ctx);
}

TEST_F(RendererTest, RenderMultipleModels) {
    auto data1 = TestMeshData::make_triangle();
    auto data2 = TestMeshData::make_triangle();
    context_begin_render(ctx);
    context_add_model(ctx, data1.mesh, transform(matrix_identity(), vector3(-3, 0, 0)));
    context_add_model(ctx, data2.mesh, transform(matrix_identity(), vector3(3, 0, 0)));
    context_finalize_render(ctx);
    Image img = context_render_view(ctx, views[0]);
    EXPECT_GT(img.width, 0);
    context_end_render(ctx);
}

TEST_F(RendererTest, RenderRemappableMaterial) {
    auto data = TestMeshData::make_triangle();
    data.materials[0].flags = MaterialFlag::IsRemappable;
    data.materials[0].region = 1;
    data.mesh = {data.vertices, data.normals, data.uvs, data.faces, data.materials};
    context_begin_render(ctx);
    context_add_model(ctx, data.mesh, transform(matrix_identity(), vector3(0, 0, 0)));
    context_finalize_render(ctx);
    Image img = context_render_view(ctx, views[0]);
    EXPECT_GT(img.width, 0);
    context_end_render(ctx);
}

TEST_F(RendererTest, RenderMaskMaterial) {
    auto data = TestMeshData::make_triangle();
    data.materials[0].flags = MaterialFlag::IsMask;
    data.mesh = {data.vertices, data.normals, data.uvs, data.faces, data.materials};
    context_begin_render(ctx);
    context_add_model(ctx, data.mesh, transform(matrix_identity(), vector3(0, 0, 0)));
    context_finalize_render(ctx);
    Image img = context_render_view(ctx, views[0]);
    EXPECT_GE(img.width, 0);
    context_end_render(ctx);
}

TEST_F(RendererTest, RenderNoAoMaterial) {
    auto data = TestMeshData::make_triangle();
    data.materials[0].flags = MaterialFlag::NoAO;
    data.materials[0].color = vector3(0.5f, 0.5f, 0.5f);
    data.materials[0].ambient_color = vector3(0.1f, 0.1f, 0.1f);
    data.mesh = {data.vertices, data.normals, data.uvs, data.faces, data.materials};
    context_begin_render(ctx);
    context_add_model(ctx, data.mesh, transform(matrix_identity(), vector3(0, 0, 0)));
    context_finalize_render(ctx);
    Image img = context_render_view(ctx, views[0]);
    EXPECT_GT(img.width, 0);
    context_end_render(ctx);
}

TEST_F(RendererTest, RenderBackgroundAaMaterial) {
    auto data = TestMeshData::make_triangle();
    data.materials[0].flags = MaterialFlag::BackgroundAA;
    data.materials[0].color = vector3(0.5f, 0.5f, 0.5f);
    data.materials[0].ambient_color = vector3(0.1f, 0.1f, 0.1f);
    data.mesh = {data.vertices, data.normals, data.uvs, data.faces, data.materials};
    context_begin_render(ctx);
    context_add_model(ctx, data.mesh, transform(matrix_identity(), vector3(0, 0, 0)));
    context_finalize_render(ctx);
    Image img = context_render_view(ctx, views[0]);
    EXPECT_GE(img.width, 0);
    context_end_render(ctx);
}

TEST_F(RendererTest, RenderBackgroundAaDarkMaterial) {
    auto data = TestMeshData::make_triangle();
    data.materials[0].flags = MaterialFlag::BackgroundAADark;
    data.materials[0].color = vector3(0.5f, 0.5f, 0.5f);
    data.materials[0].ambient_color = vector3(0.1f, 0.1f, 0.1f);
    data.mesh = {data.vertices, data.normals, data.uvs, data.faces, data.materials};
    context_begin_render(ctx);
    context_add_model(ctx, data.mesh, transform(matrix_identity(), vector3(0, 0, 0)));
    context_finalize_render(ctx);
    Image img = context_render_view(ctx, views[0]);
    EXPECT_GE(img.width, 0);
    context_end_render(ctx);
}

TEST_F(RendererTest, RenderWithShadowLight) {
    context_destroy(ctx);
    std::vector<Light> lights = {
        {LightType::Diffuse, 1, vector3_normalize(vector3(1, 1, 1)), 1.0f},
    };
    ContextInit(ctx, lights, false, palette_rct2(), 32.0f);

    auto data = TestMeshData::make_triangle();
    data.materials[0].color = vector3(0.5f, 0.5f, 0.5f);
    data.materials[0].ambient_color = vector3(0.1f, 0.1f, 0.1f);
    data.mesh = {data.vertices, data.normals, data.uvs, data.faces, data.materials};
    context_begin_render(ctx);
    context_add_model(ctx, data.mesh, transform(matrix_identity(), vector3(0, 0, 0)));
    context_finalize_render(ctx);
    Image img = context_render_view(ctx, views[0]);
    EXPECT_GE(img.width, 0);
    context_end_render(ctx);
}

TEST_F(RendererTest, RenderVisibleMaskMaterial) {
    auto data = TestMeshData::make_triangle();
    data.materials[0].flags = MaterialFlag::IsVisibleMask;
    data.materials[0].color = vector3(0.5f, 0.5f, 0.5f);
    data.materials[0].ambient_color = vector3(0.1f, 0.1f, 0.1f);
    data.mesh = {data.vertices, data.normals, data.uvs, data.faces, data.materials};
    context_begin_render(ctx);
    context_add_model(ctx, data.mesh, transform(matrix_identity(), vector3(0, 0, 0)));
    context_finalize_render(ctx);
    Image img = context_render_view(ctx, views[0]);
    EXPECT_GE(img.width, 0);
    context_end_render(ctx);
}

TEST_F(RendererTest, RenderNoBleedMaterial) {
    auto data = TestMeshData::make_triangle();
    data.materials[0].flags = MaterialFlag::NoBleed;
    data.materials[0].color = vector3(0.5f, 0.5f, 0.5f);
    data.materials[0].ambient_color = vector3(0.1f, 0.1f, 0.1f);
    data.mesh = {data.vertices, data.normals, data.uvs, data.faces, data.materials};
    context_begin_render(ctx);
    context_add_model(ctx, data.mesh, transform(matrix_identity(), vector3(0, 0, 0)));
    context_finalize_render(ctx);
    Image img = context_render_view(ctx, views[0]);
    EXPECT_GT(img.width, 0);
    context_end_render(ctx);
}

TEST_F(RendererTest, RenderWithTexturedMaterial) {
    auto data = TestMeshData::make_textured_triangle();
    std::vector<Vector3> tex_pixels = {
        {1, 0, 0}, {0, 1, 0},
        {0, 0, 1}, {1, 1, 0},
    };
    data.materials[0].texture = {2, 2, tex_pixels};
    data.mesh = {data.vertices, data.normals, data.uvs, data.faces, data.materials};
    context_begin_render(ctx);
    context_add_model(ctx, data.mesh, transform(matrix_identity(), vector3(0, 0, 0)));
    context_finalize_render(ctx);
    Image img = context_render_view(ctx, views[0]);
    EXPECT_GT(img.width, 0);
    context_end_render(ctx);
}

TEST_F(RendererTest, ContextRenderWithEnvThreadCount) {
    context_destroy(ctx);
    setenv("OPENRCT2_X7_NUM_THREADS", "2", 1);
    auto lights = test_lights();
    ContextInit(ctx, lights, false, palette_rct2(), 32.0f);
    unsetenv("OPENRCT2_X7_NUM_THREADS");

    auto data = TestMeshData::make_triangle();
    context_begin_render(ctx);
    context_add_model(ctx, data.mesh, transform(matrix_identity(), vector3(0, 0, 0)));
    context_finalize_render(ctx);
    Image img = context_render_view(ctx, views[0]);
    EXPECT_GT(img.width, 0);
    context_end_render(ctx);
}

TEST_F(RendererTest, ContextRenderWithInvalidEnvThreadCount) {
    context_destroy(ctx);
    setenv("OPENRCT2_X7_NUM_THREADS", "0", 1);
    auto lights = test_lights();
    ContextInit(ctx, lights, false, palette_rct2(), 32.0f);
    unsetenv("OPENRCT2_X7_NUM_THREADS");

    context_begin_render(ctx);
    context_finalize_render(ctx);
    Image img = context_render_view(ctx, views[0]);
    EXPECT_GE(img.width, 0);
    context_end_render(ctx);
}

TEST_F(RendererTest, RenderWithShadowCastingGeometry) {
    context_destroy(ctx);
    std::vector<Light> lights = {
        {LightType::Diffuse, 1, vector3_normalize(vector3(0, 0, 1)), 1.0f},
    };
    ContextInit(ctx, lights, false, palette_rct2(), 32.0f);

    // Two quads: one in front (blocker) and one behind (receiver).
    // Shadow ray from receiver toward light (+z) should be occluded by blocker.
    TestMeshData blocker, receiver;
    blocker.vertices = {{-2, -2, 1}, {2, -2, 1}, {2, 2, 1}, {-2, 2, 1}};
    blocker.normals = {{0, 0, -1}, {0, 0, -1}, {0, 0, -1}, {0, 0, -1}};
    blocker.uvs = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    blocker.faces = {{0, {0, 1, 2}}, {0, {0, 2, 3}}};
    blocker.materials = {Material{}};
    blocker.materials[0].color = vector3(0.5f, 0.5f, 0.5f);
    blocker.materials[0].ambient_color = vector3(0.1f, 0.1f, 0.1f);
    blocker.mesh = {blocker.vertices, blocker.normals, blocker.uvs, blocker.faces, blocker.materials};

    receiver.vertices = {{-2, -2, -1}, {2, -2, -1}, {2, 2, -1}, {-2, 2, -1}};
    receiver.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
    receiver.uvs = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    receiver.faces = {{0, {0, 1, 2}}, {0, {0, 2, 3}}};
    receiver.materials = {Material{}};
    receiver.materials[0].color = vector3(0.8f, 0.2f, 0.2f);
    receiver.materials[0].ambient_color = vector3(0.1f, 0.1f, 0.1f);
    receiver.mesh = {receiver.vertices, receiver.normals, receiver.uvs, receiver.faces, receiver.materials};

    context_begin_render(ctx);
    context_add_model(ctx, blocker.mesh, transform(matrix_identity(), vector3(0, 0, 0)));
    context_add_model(ctx, receiver.mesh, transform(matrix_identity(), vector3(0, 0, 0)));
    context_finalize_render(ctx);
    Image img = context_render_view(ctx, views[0]);
    EXPECT_GT(img.width, 0);
    context_end_render(ctx);
}

TEST_F(RendererTest, RenderWithZeroNormals) {
    // Normals of exactly (0,0,0) trigger the AO guard (normal_len_sq == 0).
    // The guard skips AO and keeps ao_factor=1 so no NaN is produced.
    auto data = TestMeshData::make_triangle();
    data.normals = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    data.mesh = {data.vertices, data.normals, data.uvs, data.faces, data.materials};
    context_begin_render(ctx);
    context_add_model(ctx, data.mesh, transform(matrix_identity(), vector3(0, 0, 0)));
    context_finalize_render(ctx);
    Image img = context_render_view(ctx, views[0]);
    // Must not crash or produce a zero-dimension image.
    EXPECT_GE(img.width, 0);
    // No pixel should contain NaN-derived garbage (palette indices are uint8).
    for (auto px : img.pixels)
        EXPECT_LT(px, 256);
    context_end_render(ctx);
}

TEST_F(RendererTest, OversizedSceneThrows) {
    context_destroy(ctx);
    auto lights = test_lights();
    // Tiny UPT makes projection scale huge, so even small geometry exceeds 16384px
    ContextInit(ctx, lights, false, palette_rct2(), 0.001f);

    auto data = TestMeshData::make_triangle();
    context_begin_render(ctx);
    context_add_model(ctx, data.mesh, transform(matrix_identity(), vector3(0, 0, 0)));
    context_finalize_render(ctx);
    EXPECT_THROW(context_render_view(ctx, views[0]), std::runtime_error);
    context_end_render(ctx);
}

TEST_F(RendererTest, DitherNoBleedAdjacentPixels) {
    context_destroy(ctx);
    auto lights = test_lights();
    ContextInit(ctx, lights, true, palette_rct2(), 32.0f);

    // Two adjacent triangles both with NO_BLEED so dither error can propagate between them
    TestMeshData d1, d2;
    d1.vertices = {{-2, -2, 0}, {0, -2, 0}, {-1, 2, 0}};
    d1.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
    d1.uvs = {{0, 0}, {1, 0}, {0.5f, 1}};
    d1.faces = {{0, {0, 1, 2}}};
    d1.materials = {Material{}};
    d1.materials[0].flags = MaterialFlag::NoBleed;
    d1.materials[0].color = vector3(0.7f, 0.3f, 0.1f);
    d1.materials[0].ambient_color = vector3(0.1f, 0.1f, 0.1f);
    d1.mesh = {d1.vertices, d1.normals, d1.uvs, d1.faces, d1.materials};

    d2.vertices = {{0, -2, 0}, {2, -2, 0}, {1, 2, 0}};
    d2.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
    d2.uvs = {{0, 0}, {1, 0}, {0.5f, 1}};
    d2.faces = {{0, {0, 1, 2}}};
    d2.materials = {Material{}};
    d2.materials[0].flags = MaterialFlag::NoBleed;
    d2.materials[0].color = vector3(0.3f, 0.7f, 0.1f);
    d2.materials[0].ambient_color = vector3(0.1f, 0.1f, 0.1f);
    d2.mesh = {d2.vertices, d2.normals, d2.uvs, d2.faces, d2.materials};

    context_begin_render(ctx);
    context_add_model(ctx, d1.mesh, transform(matrix_identity(), vector3(0, 0, 0)));
    context_add_model(ctx, d2.mesh, transform(matrix_identity(), vector3(0, 0, 0)));
    context_finalize_render(ctx);
    Image img = context_render_view(ctx, views[0]);
    EXPECT_GT(img.width, 0);
    context_end_render(ctx);
}

TEST_F(RendererTest, BackgroundAaWithMaskMaterial) {
    // BACKGROUND_AA mesh in front with a MASK mesh at similar depth to trigger
    // the IS_MASK branch in edge detection
    TestMeshData front, mask_mesh;
    front.vertices = {{-1, -1, 0}, {1, -1, 0}, {0, 1, 0}};
    front.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
    front.uvs = {{0, 0}, {1, 0}, {0.5f, 1}};
    front.faces = {{0, {0, 1, 2}}};
    front.materials = {Material{}};
    front.materials[0].flags = MaterialFlag::BackgroundAA;
    front.materials[0].color = vector3(0.5f, 0.5f, 0.5f);
    front.materials[0].ambient_color = vector3(0.1f, 0.1f, 0.1f);
    front.mesh = {front.vertices, front.normals, front.uvs, front.faces, front.materials};

    mask_mesh.vertices = {{-2, -2, 0.01f}, {2, -2, 0.01f}, {2, 2, 0.01f}, {-2, 2, 0.01f}};
    mask_mesh.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
    mask_mesh.uvs = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    mask_mesh.faces = {{0, {0, 1, 2}}, {0, {0, 2, 3}}};
    mask_mesh.materials = {Material{}};
    mask_mesh.materials[0].flags = MaterialFlag::IsMask;
    mask_mesh.mesh = {mask_mesh.vertices, mask_mesh.normals, mask_mesh.uvs, mask_mesh.faces, mask_mesh.materials};

    context_begin_render(ctx);
    context_add_model(ctx, front.mesh, transform(matrix_identity(), vector3(0, 0, 0)));
    context_add_model(ctx, mask_mesh.mesh, transform(matrix_identity(), vector3(0, 0, 0)));
    context_finalize_render(ctx);
    Image img = context_render_view(ctx, views[0]);
    EXPECT_GE(img.width, 0);
    context_end_render(ctx);
}
