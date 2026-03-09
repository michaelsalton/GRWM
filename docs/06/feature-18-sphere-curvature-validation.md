# Feature 18: Sphere Curvature Validation

## Context

A sphere of radius R has uniform mean curvature 1/R everywhere. This provides an analytical ground truth to validate the cotangent Laplacian implementation.

## Requirements

1. Generate or load a sphere mesh of known radius R
2. Run `compute_curvature()` on it
3. Compute mean absolute error: `MAE = (1/V) × Σ |H(v) - 1/R|`
4. Pass if `MAE < 0.05 × (1/R)` (5% tolerance)
5. Report per-vertex statistics (min, max, mean, stddev)

## Files Modified

- `src/validate.cu` — implement `validate_curvature()` for sphere case

## Implementation Details

### Sphere Generation

Generate an icosphere by subdividing an icosahedron. At subdivision level 4, this produces ~2,562 vertices and ~5,120 faces — sufficient for validation.

```cpp
// Procedural icosphere generation
// Start with 12 vertices and 20 faces of regular icosahedron
// Subdivide each face into 4 triangles by inserting edge midpoints
// Project new vertices onto sphere of radius R
```

Alternatively, load a pre-made sphere OBJ file.

### Validation Logic

```cpp
bool validate_curvature(const MeshData& mesh, const std::vector<float>& curvature) {
    // Detect sphere: check if all vertices are equidistant from centroid
    float3 centroid = compute_centroid(mesh);
    float R = average_distance_to_centroid(mesh, centroid);

    // Check variance of distances — if high, not a sphere
    if (distance_variance > threshold) {
        printf("Mesh is not a sphere, skipping curvature validation\n");
        return true;  // not applicable, not a failure
    }

    float expected = 1.0f / R;
    float mae = 0.0f;
    for (uint32_t i = 0; i < mesh.vertex_count; ++i) {
        mae += fabsf(curvature[i] - expected);
    }
    mae /= mesh.vertex_count;

    printf("Sphere validation: R=%.4f, expected H=%.4f, MAE=%.6f (%.2f%%)\n",
           R, expected, mae, 100.0f * mae / expected);

    return mae < 0.05f * expected;
}
```

### Cylinder Validation (Secondary)

For a cylinder of radius R:
- Curved surface: H ≈ 1/(2R)
- Flat caps: H ≈ 0

This can be added as a secondary check by detecting cylindrical geometry (vertices on a circle at constant z-intervals).

## Acceptance Criteria

- [ ] Sphere validation passes with MAE < 5% of 1/R
- [ ] Validation output prints R, expected curvature, actual MAE, and pass/fail
- [ ] Works for different sphere radii (R=1, R=5, R=0.1)
- [ ] Gracefully skips non-sphere meshes

## Dependencies

- Feature 10 (curvature computation must be implemented)
