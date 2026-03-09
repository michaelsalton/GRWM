# Epic 3: Curvature Computation (Stage 1)

## Purpose

Compute per-vertex mean curvature using the cotangent Laplacian, a standard discrete differential geometry operator. The output is a `float` buffer of length V that Gravel's task shader uses to modulate element density — high-curvature regions receive more elements than flat regions.

## Mathematical Background

Mean curvature at vertex v with one-ring neighbors v_j:

```
H(v) = (1 / 2A) × Σ_j (cot α_ij + cot β_ij) × (v - v_j)
```

Where:
- A = mixed Voronoi area around v
- α_ij, β_ij = angles opposite to edge (v, v_j) in the two adjacent triangles
- Scalar curvature = |H(v)|

This is equivalent to a sparse matrix-vector product **L × P** where L is the cotangent weight matrix (sparse, ~6-8 nonzeros per row) and P is the vertex position matrix.

## Pipeline

```
Vertex positions + Face indices
        │
        ├──→ BuildCotangentWeights kernel ──→ COO sparse matrix
        │                                          │
        │                            cuSPARSE COO→CSR conversion
        │                                          │
        │                            cusparseSpMM (L × P) ──→ Laplacian vectors (float3[V])
        │
        ├──→ ComputeVoronoiAreas kernel ──→ Area buffer (float[V])
        │
        └──→ ComputeCurvatureMagnitude kernel ──→ Curvature buffer (float[V])
                     (uses Laplacian vectors + areas)
```

## Features

| # | Feature | Focus |
|---|---------|-------|
| 06 | [CUDA device helpers](feature-06-cuda-device-helpers.md) | cotangent, vector math utilities |
| 07 | [Cotangent weight assembly](feature-07-cotangent-weight-assembly.md) | BuildCotangentWeights kernel |
| 08 | [cuSPARSE SpMV pipeline](feature-08-cusparse-spmv.md) | COO→CSR + sparse matrix-vector product |
| 09 | [Voronoi area computation](feature-09-voronoi-area.md) | ComputeVoronoiAreas kernel |
| 10 | [Curvature host orchestration](feature-10-curvature-host-pipeline.md) | compute_curvature() end-to-end |

## Success Criteria

- [ ] Sphere of radius R: mean curvature uniformly ≈ 1/R (MAE < 5% of 1/R)
- [ ] Cylinder of radius R: curved surface ≈ 1/(2R), flat caps ≈ 0
- [ ] Output buffer is byte-identical between runs (deterministic)
- [ ] Performance: faster than CPU reference for meshes > 50k vertices

## Dependencies

- Epic 1 (mesh data and output I/O)

## Downstream Dependents

- Epic 5 (Stage 3 uses curvature values for slot priority scoring)
- Epic 6 (validation tests compare against analytical ground truth)
