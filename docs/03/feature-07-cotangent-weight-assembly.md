# Feature 07: Cotangent Weight Assembly Kernel

## Context

The cotangent Laplacian is represented as a sparse matrix L where `L[i][j]` contains the sum of cotangent weights for the edge (i, j). This kernel assembles the matrix in COO (coordinate) format, with one thread per triangle face computing the three cotangent weights for the face's edges and atomically accumulating them.

## Requirements

1. One CUDA thread per face
2. Compute cotangent weights for all three edges of each triangle
3. Atomically accumulate weights into COO sparse matrix (multiple faces share edges)
4. Handle edge indexing — each undirected edge needs a consistent index
5. Also accumulate negative diagonal entries (L[i][i] = -Σ_j w_ij)

## Files Modified

- `src/stage1_curvature.cu` — implement `BuildCotangentWeights` kernel

## Implementation Details

### COO Format

The sparse matrix is stored as three parallel arrays:
- `coo_row[k]` — row index (source vertex)
- `coo_col[k]` — column index (neighbor vertex)
- `coo_values[k]` — weight value

For each edge (vi, vj) in a triangle, the cotangent weight contributes to four COO entries:
- L[vi][vj] += cot(angle opposite to edge)
- L[vj][vi] += cot(angle opposite to edge)  (symmetric)
- L[vi][vi] -= cot(angle opposite to edge)  (diagonal)
- L[vj][vj] -= cot(angle opposite to edge)  (diagonal)

### Edge Indexing Strategy

Pre-allocate COO arrays with known maximum size:
- Each face contributes to 6 off-diagonal entries (3 edges × 2 directions) and 6 diagonal updates
- Total COO entries ≤ 6F off-diagonal + V diagonal
- Alternative: pre-compute edge indices from half-edge mesh and use those

### Kernel

```cuda
__global__ void BuildCotangentWeights(
    const float3* positions,
    const uint3*  faces,
    float*        coo_values,      // pre-allocated, initialized to 0
    uint32_t*     coo_row,
    uint32_t*     coo_col,
    uint32_t      face_count)
{
    uint32_t fid = blockIdx.x * blockDim.x + threadIdx.x;
    if (fid >= face_count) return;

    uint3 f = faces[fid];
    float3 v0 = positions[f.x];
    float3 v1 = positions[f.y];
    float3 v2 = positions[f.z];

    float cot0 = cotangent(v1, v0, v2);  // angle at v0, opposite edge (v1,v2)
    float cot1 = cotangent(v2, v1, v0);  // angle at v1, opposite edge (v0,v2)
    float cot2 = cotangent(v0, v2, v1);  // angle at v2, opposite edge (v0,v1)

    // Edge (v1,v2): weight = cot0
    atomicAdd(&values_at[edge(f.y, f.z)], cot0);
    // Edge (v0,v2): weight = cot1
    atomicAdd(&values_at[edge(f.x, f.z)], cot1);
    // Edge (v0,v1): weight = cot2
    atomicAdd(&values_at[edge(f.x, f.y)], cot2);
}
```

### Atomic Contention

Each edge is shared by exactly 2 faces (on a manifold mesh), so each COO entry receives exactly 2 atomic adds. Contention is low — this is not a performance bottleneck.

### Block Configuration

```cpp
int threads = 256;
int blocks = (face_count + threads - 1) / threads;
BuildCotangentWeights<<<blocks, threads>>>(...);
```

## Acceptance Criteria

- [ ] COO matrix has correct number of nonzero entries
- [ ] Weight values are symmetric: w(i,j) == w(j,i)
- [ ] Cotangent values are clamped to [-10, 10]
- [ ] Equilateral triangle mesh: all weights ≈ cot(60°) ≈ 0.577
- [ ] No CUDA errors from atomic operations

## Dependencies

- Feature 06 (cotangent device function)
- Feature 02 (mesh data on host)
