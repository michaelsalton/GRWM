# Feature 09: Voronoi Area Computation

## Context

The mean curvature formula divides the Laplacian vector by twice the mixed Voronoi area around each vertex. This area normalizes the curvature estimate so it is independent of mesh resolution. Without it, vertices in dense regions would have artificially low curvature.

## Requirements

1. One CUDA thread per face
2. Compute mixed Voronoi area contribution for each vertex of the triangle
3. Atomically accumulate area to vertex buffer (shared vertices across faces)
4. Handle obtuse triangles with the Meyer et al. correction

## Files Modified

- `src/stage1_curvature.cu` — implement `ComputeVoronoiAreas` kernel

## Implementation Details

### Mixed Voronoi Area (Meyer et al. 2002)

For a triangle with vertices (v0, v1, v2):

**Non-obtuse triangle:**
Each vertex gets its Voronoi area:
```
A_voronoi(v0) = (1/8) × (|v0-v2|² × cot(α₁) + |v0-v1|² × cot(α₂))
```
where α₁ is the angle at v1 and α₂ is the angle at v2.

**Obtuse triangle:**
The obtuse vertex gets half the triangle area. The other two vertices each get one quarter.

### Simplified Approach

For implementation simplicity, a common approximation accumulates one-third of each triangle's area to each vertex. This is less accurate than the full mixed Voronoi area but is simpler and sufficient for visualization-quality curvature:

```cuda
__global__ void ComputeVoronoiAreas(
    const float3* positions,
    const uint3*  faces,
    float*        vertex_areas,
    uint32_t      face_count)
{
    uint32_t fid = blockIdx.x * blockDim.x + threadIdx.x;
    if (fid >= face_count) return;

    uint3 f = faces[fid];
    float3 v0 = positions[f.x];
    float3 v1 = positions[f.y];
    float3 v2 = positions[f.z];

    // Edge vectors
    float3 e1 = {v1.x - v0.x, v1.y - v0.y, v1.z - v0.z};
    float3 e2 = {v2.x - v0.x, v2.y - v0.y, v2.z - v0.z};

    // Cross product magnitude = 2 × triangle area
    float3 cross = {
        e1.y * e2.z - e1.z * e2.y,
        e1.z * e2.x - e1.x * e2.z,
        e1.x * e2.y - e1.y * e2.x
    };
    float area = 0.5f * sqrtf(cross.x*cross.x + cross.y*cross.y + cross.z*cross.z);

    // Each vertex gets 1/3 of the triangle area
    float third = area / 3.0f;
    atomicAdd(&vertex_areas[f.x], third);
    atomicAdd(&vertex_areas[f.y], third);
    atomicAdd(&vertex_areas[f.z], third);
}
```

### Full Mixed Voronoi Area (Recommended)

For higher accuracy, implement the obtuse triangle correction from Meyer et al.:

```
if triangle is non-obtuse:
    A(v_i) += voronoi_area(v_i)  // cotangent-based formula
else if angle at v_i is obtuse:
    A(v_i) += triangle_area / 2
else:
    A(v_i) += triangle_area / 4
```

### Initialization

The `vertex_areas` buffer must be initialized to zero before the kernel launch:
```cpp
cudaMemset(d_vertex_areas, 0, vertex_count * sizeof(float));
```

## Acceptance Criteria

- [ ] Sum of all vertex areas ≈ total mesh surface area
- [ ] Uniform sphere: all vertex areas approximately equal (varies with mesh regularity)
- [ ] No negative or zero areas for valid triangles
- [ ] Degenerate (zero-area) triangles contribute 0 to all vertices

## Dependencies

- Feature 06 (cotangent function, if using full Voronoi formula)
- Mesh data uploaded to device (part of Feature 10 orchestration)
