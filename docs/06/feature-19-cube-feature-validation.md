# Feature 19: Cube Feature Edge Validation

## Context

A unit cube has 12 edges, all with 90° dihedral angles. At the default 30° threshold, every edge should be flagged as a feature edge, and every face should have feature_flag = 1.

## Requirements

1. Generate or load a cube mesh (8 vertices, 12 triangular faces)
2. Run `detect_feature_edges()` with default threshold (30°)
3. Verify all faces are flagged (flag = 1)
4. Report pass/fail with details

## Files Modified

- `src/validate.cu` — add cube validation logic

## Implementation Details

### Cube Mesh

A unit cube centered at origin with 8 vertices and 6 quad faces, triangulated into 12 triangles:

```cpp
// 8 vertices of unit cube
float positions[] = {
    -0.5f, -0.5f, -0.5f,   // 0
     0.5f, -0.5f, -0.5f,   // 1
     0.5f,  0.5f, -0.5f,   // 2
    -0.5f,  0.5f, -0.5f,   // 3
    -0.5f, -0.5f,  0.5f,   // 4
     0.5f, -0.5f,  0.5f,   // 5
     0.5f,  0.5f,  0.5f,   // 6
    -0.5f,  0.5f,  0.5f,   // 7
};

// 12 triangles (2 per quad face)
uint32_t indices[] = {
    0,1,2, 0,2,3,  // front
    4,6,5, 4,7,6,  // back
    0,4,5, 0,5,1,  // bottom
    2,6,7, 2,7,3,  // top
    0,3,7, 0,7,4,  // left
    1,5,6, 1,6,2,  // right
};
```

### Validation Logic

```cpp
// All 12 faces should be flagged
uint32_t flagged = 0;
for (uint32_t i = 0; i < face_count; ++i) {
    if (features[i]) flagged++;
}
printf("Cube validation: %u/%u faces flagged (expected %u)\n",
       flagged, face_count, face_count);
return flagged == face_count;
```

### Subdivided Sphere Counter-Check

A smooth sphere at 30° threshold should have zero flagged faces. This can be a secondary validation:

```cpp
auto sphere_features = detect_feature_edges(sphere_mesh, 30.0f);
uint32_t sphere_flagged = std::count(sphere_features.begin(), sphere_features.end(), 1);
printf("Sphere validation: %u faces flagged (expected 0)\n", sphere_flagged);
```

## Acceptance Criteria

- [ ] All 12 cube faces are flagged
- [ ] Smooth sphere has 0 flagged faces
- [ ] Validation prints clear pass/fail message
- [ ] Works with default threshold (30°) and custom thresholds

## Dependencies

- Feature 13 (feature edge detection must be implemented)
