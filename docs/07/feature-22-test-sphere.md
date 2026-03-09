# Feature 22: test_sphere Executable

## Context

Automated test that generates a procedural icosphere, runs the curvature pipeline, and verifies results against the analytical ground truth H = 1/R.

## Requirements

1. Generate icosphere of radius R at subdivision level 4 (~2,562 vertices, ~5,120 faces)
2. Run `compute_curvature()` on the generated mesh
3. Compute MAE against expected curvature 1/R
4. Exit with code 0 if MAE < 5% of 1/R, exit with code 1 otherwise
5. Print statistics: R, expected H, MAE, min/max/mean curvature

## Files Modified

- `tests/test_sphere.cpp` — full implementation

## Implementation Details

### Icosphere Generation

Start with a regular icosahedron (12 vertices, 20 faces) and subdivide:

```cpp
struct IcoSphere {
    std::vector<float> positions;
    std::vector<uint32_t> indices;
};

IcoSphere generate_icosphere(float radius, int subdivisions) {
    // 1. Create 12 vertices of icosahedron
    float t = (1.0f + sqrtf(5.0f)) / 2.0f;  // golden ratio
    // Vertices: (±1, ±t, 0), (0, ±1, ±t), (±t, 0, ±1)
    // Normalize to radius R

    // 2. Create 20 triangular faces

    // 3. For each subdivision level:
    //    For each triangle:
    //      Insert midpoints on each edge (reuse existing midpoints)
    //      Replace 1 triangle with 4 sub-triangles
    //    Project all new vertices onto sphere of radius R

    // After level 4: V ≈ 10*4^4 + 2 = 2562, F = 20*4^4 = 5120
}
```

### Test Logic

```cpp
int main() {
    float R = 1.0f;
    auto sphere = generate_icosphere(R, 4);

    grwm::MeshData mesh;
    mesh.positions = sphere.positions;
    mesh.indices = sphere.indices;
    mesh.vertex_count = sphere.positions.size() / 3;
    mesh.face_count = sphere.indices.size() / 3;

    auto curvature = grwm::compute_curvature(mesh);
    if (curvature.empty()) {
        printf("FAIL: compute_curvature returned empty\n");
        return 1;
    }

    float expected = 1.0f / R;
    float mae = 0.0f, min_c = FLT_MAX, max_c = 0.0f;
    for (float c : curvature) {
        mae += fabsf(c - expected);
        min_c = fminf(min_c, c);
        max_c = fmaxf(max_c, c);
    }
    mae /= curvature.size();

    printf("Sphere test: R=%.2f, expected=%.4f\n", R, expected);
    printf("  min=%.4f, max=%.4f, mean=%.4f, MAE=%.6f (%.2f%%)\n",
           min_c, max_c, mae + expected, mae, 100.0f * mae / expected);

    bool pass = mae < 0.05f * expected;
    printf("Result: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
```

## Acceptance Criteria

- [ ] Generates valid icosphere mesh (no degenerate triangles)
- [ ] Curvature computation succeeds (non-empty result)
- [ ] MAE < 5% of 1/R
- [ ] Exit code 0 on pass, 1 on fail
- [ ] Runs in under 10 seconds (including GPU warmup)

## Dependencies

- Feature 10 (curvature pipeline must be implemented)
