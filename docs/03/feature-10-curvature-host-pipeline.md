# Feature 10: Curvature Host Orchestration

## Context

This feature ties together all Stage 1 components into the `compute_curvature()` host function. It manages device memory allocation, kernel launches, cuSPARSE calls, and result copy-back.

## Requirements

1. Implement the full `compute_curvature(const MeshData& mesh) -> vector<float>` function
2. Manage all device memory allocations and frees
3. Launch kernels in correct order with proper synchronization
4. Handle CUDA and cuSPARSE error checking throughout
5. Return curvature buffer to host, or empty vector on failure

## Files Modified

- `src/stage1_curvature.cu` — implement `compute_curvature()` body

## Implementation Details

### Pipeline Sequence

```
1. Allocate device memory
   - d_positions:    float3[V]        (vertex positions)
   - d_faces:        uint3[F]         (face indices)
   - d_coo_values:   float[nnz_max]   (sparse matrix values)
   - d_coo_row:      uint32[nnz_max]  (sparse matrix row indices)
   - d_coo_col:      uint32[nnz_max]  (sparse matrix column indices)
   - d_laplacian:    float[V*3]       (output Laplacian vectors)
   - d_vertex_areas: float[V]         (Voronoi areas)
   - d_curvature:    float[V]         (final output)

2. Copy mesh data to device
   - positions: reinterpret flat float array as float3 array
   - faces: reinterpret flat uint32 array as uint3 array

3. Initialize COO arrays and areas to zero

4. Launch BuildCotangentWeights<<<blocks, 256>>>(...)
   - Populates COO sparse matrix

5. cuSPARSE: COO → CSR conversion
   - cusparseXcoo2csr()

6. cuSPARSE: SpMM (L × P)
   - cusparseSpMM() → d_laplacian

7. Launch ComputeVoronoiAreas<<<blocks, 256>>>(...)
   - Populates d_vertex_areas

8. Launch ComputeCurvatureMagnitude<<<blocks, 256>>>(...)
   - d_curvature[v] = length(d_laplacian[v]) / (2.0 * d_vertex_areas[v])

9. Copy d_curvature back to host

10. Free all device memory + cuSPARSE handles
```

### NNZ Estimation

Maximum number of nonzero entries in the cotangent weight matrix:
- Each face contributes weights for 3 edges, each edge appears twice (symmetric)
- Off-diagonal: up to 6F entries (before deduplication; cuSPARSE handles duplicates in COO)
- Diagonal: V entries
- Conservative estimate: `nnz_max = 6 * face_count + vertex_count`

### ComputeCurvatureMagnitude Kernel

```cuda
__global__ void ComputeCurvatureMagnitude(
    const float* laplacian,    // V*3 floats (Lx, Ly, Lz per vertex)
    const float* vertex_areas,
    float*       curvature_out,
    uint32_t     vertex_count)
{
    uint32_t vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= vertex_count) return;

    float lx = laplacian[vid * 3 + 0];
    float ly = laplacian[vid * 3 + 1];
    float lz = laplacian[vid * 3 + 2];
    float len = sqrtf(lx*lx + ly*ly + lz*lz);
    float area = vertex_areas[vid];

    curvature_out[vid] = (area > 1e-10f) ? len / (2.0f * area) : 0.0f;
}
```

### Error Handling

Wrap every CUDA and cuSPARSE call with error checking. On failure, free all allocated memory and return an empty vector.

### Memory Usage

For a 1M vertex mesh:
- Positions: 12 MB
- Faces: ~8 MB (assuming ~2M faces for manifold mesh)
- COO arrays: ~96 MB (6 × 2M × 4 bytes × 3 arrays)
- Laplacian: 12 MB
- Areas + curvature: 8 MB
- **Total: ~136 MB**

## Acceptance Criteria

- [ ] `compute_curvature()` returns a non-empty vector of length `vertex_count`
- [ ] No CUDA memory leaks (all allocations freed)
- [ ] Sphere validation passes (MAE < 5% of 1/R)
- [ ] Curvature values are non-negative (magnitude)
- [ ] Zero-area vertices get curvature = 0 (no division by zero)

## Dependencies

- Feature 07 (cotangent weight kernel)
- Feature 08 (cuSPARSE SpMV)
- Feature 09 (Voronoi area kernel)
- Feature 02 (mesh data loaded)
