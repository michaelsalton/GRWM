# Feature 05: GPU Adjacency Data Upload

## Context

The feature edge detection kernel runs on the GPU and needs the half-edge adjacency data. The CPU-built `HalfEdgeMesh` must be uploaded to device memory as flat `uint32_t` arrays that the CUDA kernel can index into.

## Requirements

1. Allocate device memory for adjacency arrays
2. Upload `edge_to_face` array (maps edge index → face index for each half-edge direction)
3. Upload `half_edge_opposite` array (maps half-edge index → opposite half-edge index)
4. Handle boundary edges (opposite = UINT32_MAX sentinel)
5. Free device memory after kernel execution

## Files Modified

- `src/stage2_features.cu` — within `detect_feature_edges()` host function

## Implementation Details

### GPU Arrays Needed

```cpp
// Device arrays for feature edge detection
uint32_t* d_half_edge_opposite;  // size: num_half_edges (3 * face_count)
uint32_t* d_edge_to_face;        // size: num_half_edges (face index per half-edge)
```

### Upload Pattern

```cpp
// Flatten HalfEdgeMesh data for GPU
std::vector<uint32_t> opposites(he_mesh.half_edges.size());
std::vector<uint32_t> he_faces(he_mesh.half_edges.size());
for (size_t i = 0; i < he_mesh.half_edges.size(); ++i) {
    opposites[i] = he_mesh.half_edges[i].opposite;
    he_faces[i]  = he_mesh.half_edges[i].face;
}

cudaMalloc(&d_half_edge_opposite, opposites.size() * sizeof(uint32_t));
cudaMalloc(&d_edge_to_face,       he_faces.size() * sizeof(uint32_t));
cudaMemcpy(d_half_edge_opposite, opposites.data(), ..., cudaMemcpyHostToDevice);
cudaMemcpy(d_edge_to_face,       he_faces.data(),  ..., cudaMemcpyHostToDevice);
```

### Memory Estimate

For a 200k face mesh:
- Half-edges: 600k × 4 bytes × 2 arrays = ~4.8 MB
- Negligible relative to slot buffer sizes

### Cleanup

```cpp
cudaFree(d_half_edge_opposite);
cudaFree(d_edge_to_face);
```

Use RAII wrappers or explicit cleanup in the host orchestration function.

## Acceptance Criteria

- [ ] Device arrays are allocated and populated without CUDA errors
- [ ] Kernel can read adjacency data and produce correct face normals
- [ ] Device memory is freed after kernel completion
- [ ] `GRWM_CUDA_CHECK` validates all CUDA API calls

## Dependencies

- Feature 04 (half-edge mesh must be built on CPU first)
