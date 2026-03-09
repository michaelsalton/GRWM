#include "preprocess.h"

#include <cstdint>
#include <cstdio>

static grwm::MeshData make_cube() {
    grwm::MeshData mesh;
    mesh.positions = {
        -0.5f, -0.5f, -0.5f,  //  0
         0.5f, -0.5f, -0.5f,  //  1
         0.5f,  0.5f, -0.5f,  //  2
        -0.5f,  0.5f, -0.5f,  //  3
        -0.5f, -0.5f,  0.5f,  //  4
         0.5f, -0.5f,  0.5f,  //  5
         0.5f,  0.5f,  0.5f,  //  6
        -0.5f,  0.5f,  0.5f,  //  7
    };
    mesh.indices = {
        0, 1, 2,  0, 2, 3,  // -Z
        4, 6, 5,  4, 7, 6,  // +Z
        0, 4, 5,  0, 5, 1,  // -Y
        2, 6, 7,  2, 7, 3,  // +Y
        0, 3, 7,  0, 7, 4,  // -X
        1, 5, 6,  1, 6, 2,  // +X
    };
    mesh.vertex_count = 8;
    mesh.face_count = 12;
    return mesh;
}

int main() {
    printf("=== Cube Feature Edge Test ===\n\n");

    auto mesh = make_cube();
    printf("Cube mesh: %u vertices, %u faces\n\n", mesh.vertex_count, mesh.face_count);

    printf("Detecting feature edges (threshold=30 deg)...\n");
    auto features = grwm::detect_feature_edges(mesh, 30.0f);

    if (features.size() != mesh.face_count) {
        printf("FAIL: expected %u face flags, got %zu\n",
               mesh.face_count, features.size());
        return 1;
    }

    uint32_t flagged = 0;
    for (auto f : features) flagged += f;

    printf("\nResults:\n");
    printf("  Faces flagged: %u / %u\n", flagged, mesh.face_count);

    bool pass = (flagged == mesh.face_count);
    printf("\nResult: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
