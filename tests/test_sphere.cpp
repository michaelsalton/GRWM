#include "preprocess.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <vector>

// --- Procedural icosphere generation ---

static void normalize_to_radius(float& x, float& y, float& z, float R) {
    float len = sqrtf(x * x + y * y + z * z);
    if (len > 1e-10f) {
        x = x / len * R;
        y = y / len * R;
        z = z / len * R;
    }
}

static uint64_t edge_key(uint32_t a, uint32_t b) {
    uint32_t lo = (a < b) ? a : b;
    uint32_t hi = (a < b) ? b : a;
    return (uint64_t(lo) << 32) | uint64_t(hi);
}

static grwm::MeshData generate_icosphere(float R, int subdivisions) {
    std::vector<float> positions;
    std::vector<uint32_t> indices;

    // Icosahedron vertices
    float t = (1.0f + sqrtf(5.0f)) / 2.0f;

    float verts[][3] = {
        {-1,  t,  0}, { 1,  t,  0}, {-1, -t,  0}, { 1, -t,  0},
        { 0, -1,  t}, { 0,  1,  t}, { 0, -1, -t}, { 0,  1, -t},
        { t,  0, -1}, { t,  0,  1}, {-t,  0, -1}, {-t,  0,  1},
    };

    for (int i = 0; i < 12; ++i) {
        float x = verts[i][0], y = verts[i][1], z = verts[i][2];
        normalize_to_radius(x, y, z, R);
        positions.push_back(x);
        positions.push_back(y);
        positions.push_back(z);
    }

    // 20 icosahedron faces
    uint32_t faces[] = {
        0,11,5,  0,5,1,   0,1,7,   0,7,10,  0,10,11,
        1,5,9,   5,11,4,  11,10,2, 10,7,6,  7,1,8,
        3,9,4,   3,4,2,   3,2,6,   3,6,8,   3,8,9,
        4,9,5,   2,4,11,  6,2,10,  8,6,7,   9,8,1,
    };
    for (int i = 0; i < 60; ++i) {
        indices.push_back(faces[i]);
    }

    // Subdivide
    for (int level = 0; level < subdivisions; ++level) {
        std::vector<uint32_t> new_indices;
        std::map<uint64_t, uint32_t> midpoint_cache;

        uint32_t num_tris = static_cast<uint32_t>(indices.size() / 3);
        for (uint32_t tri = 0; tri < num_tris; ++tri) {
            uint32_t v0 = indices[tri * 3 + 0];
            uint32_t v1 = indices[tri * 3 + 1];
            uint32_t v2 = indices[tri * 3 + 2];

            // Get or create midpoint for each edge
            uint32_t mids[3];
            uint32_t edge_verts[3][2] = {{v0,v1}, {v1,v2}, {v2,v0}};

            for (int e = 0; e < 3; ++e) {
                uint64_t key = edge_key(edge_verts[e][0], edge_verts[e][1]);
                auto it = midpoint_cache.find(key);
                if (it != midpoint_cache.end()) {
                    mids[e] = it->second;
                } else {
                    uint32_t a = edge_verts[e][0], b = edge_verts[e][1];
                    float mx = (positions[a*3+0] + positions[b*3+0]) * 0.5f;
                    float my = (positions[a*3+1] + positions[b*3+1]) * 0.5f;
                    float mz = (positions[a*3+2] + positions[b*3+2]) * 0.5f;
                    normalize_to_radius(mx, my, mz, R);

                    uint32_t idx = static_cast<uint32_t>(positions.size() / 3);
                    positions.push_back(mx);
                    positions.push_back(my);
                    positions.push_back(mz);
                    midpoint_cache[key] = idx;
                    mids[e] = idx;
                }
            }

            // 4 sub-triangles
            new_indices.push_back(v0);      new_indices.push_back(mids[0]); new_indices.push_back(mids[2]);
            new_indices.push_back(v1);      new_indices.push_back(mids[1]); new_indices.push_back(mids[0]);
            new_indices.push_back(v2);      new_indices.push_back(mids[2]); new_indices.push_back(mids[1]);
            new_indices.push_back(mids[0]); new_indices.push_back(mids[1]); new_indices.push_back(mids[2]);
        }

        indices = new_indices;
    }

    grwm::MeshData mesh;
    mesh.positions = positions;
    mesh.indices = indices;
    mesh.vertex_count = static_cast<uint32_t>(positions.size() / 3);
    mesh.face_count = static_cast<uint32_t>(indices.size() / 3);
    return mesh;
}

int main() {
    printf("=== Sphere Curvature Test ===\n\n");

    float R = 1.0f;
    int subdivisions = 4;

    printf("Generating icosphere: R=%.2f, subdivisions=%d\n", R, subdivisions);
    auto mesh = generate_icosphere(R, subdivisions);
    printf("  Vertices: %u, Faces: %u\n\n", mesh.vertex_count, mesh.face_count);

    printf("Computing curvature...\n");
    auto curvature = grwm::compute_curvature(mesh);
    if (curvature.empty()) {
        printf("FAIL: compute_curvature returned empty\n");
        return 1;
    }

    float expected = 1.0f / R;
    float mae = 0.0f;
    float min_c = curvature[0], max_c = curvature[0], sum_c = 0.0f;
    for (uint32_t i = 0; i < curvature.size(); ++i) {
        float c = curvature[i];
        mae += fabsf(c - expected);
        min_c = std::min(min_c, c);
        max_c = std::max(max_c, c);
        sum_c += c;
    }
    mae /= curvature.size();
    float mean_c = sum_c / curvature.size();
    float relative_error = mae / expected;

    printf("\nResults:\n");
    printf("  Expected H = %.4f (1/R)\n", expected);
    printf("  Min    = %.4f\n", min_c);
    printf("  Max    = %.4f\n", max_c);
    printf("  Mean   = %.4f\n", mean_c);
    printf("  MAE    = %.6f (%.2f%%)\n", mae, relative_error * 100.0f);

    bool pass = relative_error < 0.05f;
    printf("\nResult: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
