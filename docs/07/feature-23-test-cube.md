# Feature 23: test_cube Executable

## Context

Automated test that generates a unit cube mesh, runs feature edge detection, and verifies all faces are flagged as adjacent to feature edges.

## Requirements

1. Generate unit cube mesh (8 vertices, 12 triangular faces)
2. Run `detect_feature_edges()` with default 30° threshold
3. Verify all 12 faces have flag = 1
4. Exit with code 0 if all pass, 1 otherwise

## Files Modified

- `tests/test_cube.cpp` — full implementation

## Implementation Details

### Cube Mesh

```cpp
grwm::MeshData make_cube() {
    grwm::MeshData mesh;
    mesh.positions = {
        -0.5f,-0.5f,-0.5f,   0.5f,-0.5f,-0.5f,   0.5f, 0.5f,-0.5f,  -0.5f, 0.5f,-0.5f,
        -0.5f,-0.5f, 0.5f,   0.5f,-0.5f, 0.5f,   0.5f, 0.5f, 0.5f,  -0.5f, 0.5f, 0.5f,
    };
    mesh.indices = {
        0,1,2, 0,2,3,  // -Z face
        4,6,5, 4,7,6,  // +Z face
        0,4,5, 0,5,1,  // -Y face
        2,6,7, 2,7,3,  // +Y face
        0,3,7, 0,7,4,  // -X face
        1,5,6, 1,6,2,  // +X face
    };
    mesh.vertex_count = 8;
    mesh.face_count = 12;
    return mesh;
}
```

### Test Logic

```cpp
int main() {
    auto mesh = make_cube();
    auto features = grwm::detect_feature_edges(mesh, 30.0f);

    if (features.size() != mesh.face_count) {
        printf("FAIL: expected %u face flags, got %zu\n",
               mesh.face_count, features.size());
        return 1;
    }

    uint32_t flagged = 0;
    for (auto f : features) flagged += f;

    printf("Cube test: %u/%u faces flagged\n", flagged, mesh.face_count);

    bool pass = (flagged == mesh.face_count);
    printf("Result: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
```

## Acceptance Criteria

- [ ] All 12 faces flagged (every face is adjacent to a 90° edge)
- [ ] Exit code 0 on pass
- [ ] Runs in under 2 seconds

## Dependencies

- Feature 13 (feature edge detection must be implemented)
