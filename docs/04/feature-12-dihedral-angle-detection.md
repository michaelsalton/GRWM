# Feature 12: Dihedral Angle Detection Kernel

## Context

A feature edge is one where the dihedral angle between its two adjacent face normals exceeds a threshold. This kernel checks every edge against the threshold and writes a per-edge flag. A second reduction kernel converts edge flags to per-face flags.

## Requirements

1. `DetectFeatureEdges` kernel: one thread per edge, compare dihedral angle to threshold
2. `ReduceEdgeFlagsToFaces` kernel: one thread per face, OR its three edge flags
3. Boundary edges (adjacent to only one face) must not be flagged
4. Use `cos_threshold` for efficient comparison (avoid `acos` in the kernel)

## Files Modified

- `src/stage2_features.cu` — implement `DetectFeatureEdges` and `ReduceEdgeFlagsToFaces` kernels

## Implementation Details

### DetectFeatureEdges

```cuda
__global__ void DetectFeatureEdges(
    const float3*   face_normals,
    const uint32_t* half_edge_opposite,
    const uint32_t* edge_to_face,
    uint8_t*        edge_flags,
    float           cos_threshold,
    uint32_t        edge_count)
{
    uint32_t eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= edge_count) return;

    uint32_t he0 = eid;   // or use edge_to_halfedge mapping
    uint32_t he1 = half_edge_opposite[he0];

    // Boundary edge — skip
    if (he1 == UINT32_MAX) {
        edge_flags[eid] = 0;
        return;
    }

    uint32_t f0 = edge_to_face[he0];
    uint32_t f1 = edge_to_face[he1];

    float3 n0 = face_normals[f0];
    float3 n1 = face_normals[f1];

    float d = n0.x*n1.x + n0.y*n1.y + n0.z*n1.z;

    // If dot product < cos_threshold, angle > threshold → feature edge
    edge_flags[eid] = (d < cos_threshold) ? 1 : 0;
}
```

### Threshold Conversion

On the host, before kernel launch:
```cpp
float threshold_rad = threshold_degrees * M_PI / 180.0f;
float cos_threshold = cosf(threshold_rad);
```

For the default 30° threshold: `cos(30°) ≈ 0.866`. If `dot(n0, n1) < 0.866`, the angle between normals exceeds 30°.

### ReduceEdgeFlagsToFaces

Each face has 3 edges. The face is flagged if any of its edges is a feature edge.

```cuda
__global__ void ReduceEdgeFlagsToFaces(
    const uint8_t*  edge_flags,
    const uint32_t* face_edge_indices,  // 3 edge indices per face, flat array
    uint8_t*        face_flags,
    uint32_t        face_count)
{
    uint32_t fid = blockIdx.x * blockDim.x + threadIdx.x;
    if (fid >= face_count) return;

    uint32_t e0 = face_edge_indices[fid * 3 + 0];
    uint32_t e1 = face_edge_indices[fid * 3 + 1];
    uint32_t e2 = face_edge_indices[fid * 3 + 2];

    face_flags[fid] = (edge_flags[e0] | edge_flags[e1] | edge_flags[e2]);
}
```

### face_edge_indices Construction

This mapping (face → its 3 edge indices) must be built during half-edge construction (Feature 04) and uploaded to the GPU. Each face's 3 half-edges correspond to 3 edges. The edge index for a half-edge can be derived from the half-edge index or stored explicitly.

## Acceptance Criteria

- [ ] Cube: all edges flagged (dihedral angle = 90° > 30° threshold)
- [ ] Smooth sphere: no edges flagged
- [ ] Cylinder: edges between flat caps and curved surface flagged; edges within curved surface not flagged
- [ ] Boundary edges produce flag = 0
- [ ] Reducing to faces: cube has all 12 faces flagged

## Dependencies

- Feature 05 (adjacency data on GPU)
- Feature 11 (face normals computed)
