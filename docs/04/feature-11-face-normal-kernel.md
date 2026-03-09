# Feature 11: Face Normal Computation Kernel

## Context

Face normals are needed by both Stage 2 (dihedral angle computation) and potentially Stage 3 (if normal-based priority scoring is added later). They are computed once and reused. Each face normal is the normalized cross product of two edge vectors.

## Requirements

1. One CUDA thread per face
2. Compute face normal as normalized cross product of two edge vectors
3. Handle degenerate triangles (zero-area) gracefully — set normal to (0,0,0)
4. Output: `float3[face_count]` device buffer

## Files Modified

- `src/stage2_features.cu` — implement `ComputeFaceNormals` kernel

## Implementation Details

```cuda
__global__ void ComputeFaceNormals(
    const float3* positions,
    const uint3*  faces,
    float3*       face_normals,
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

    // Cross product
    float3 n = {
        e1.y * e2.z - e1.z * e2.y,
        e1.z * e2.x - e1.x * e2.z,
        e1.x * e2.y - e1.y * e2.x
    };

    // Normalize
    float len = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
    if (len > 1e-10f) {
        n.x /= len;
        n.y /= len;
        n.z /= len;
    } else {
        n = {0.0f, 0.0f, 0.0f};  // degenerate triangle
    }

    face_normals[fid] = n;
}
```

### Performance

This is an embarrassingly parallel kernel with no atomic operations or shared memory needed. Each thread reads 3 vertices (9 floats) and writes 1 normal (3 floats). Memory-bound for large meshes.

Block size: 256 threads. Grid size: `(face_count + 255) / 256`.

## Acceptance Criteria

- [ ] Unit cube: 6 unique normals (±x, ±y, ±z), each shared by 2 triangular faces
- [ ] Normals are unit length (|n| ≈ 1.0) for non-degenerate triangles
- [ ] Degenerate triangles produce (0,0,0) normal without crashes
- [ ] Consistent winding order (normals point outward for standard OBJ convention)

## Dependencies

- Mesh positions and faces uploaded to device (part of Feature 13 orchestration)
