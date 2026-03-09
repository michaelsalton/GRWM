# Epic 2: Half-Edge Data Structure

## Purpose

Build the half-edge adjacency structure needed by Stage 2 (feature edge detection). The half-edge table maps every directed edge to its opposite half-edge, enabling O(1) lookup of the two faces adjacent to any edge. This structure is built on the CPU and uploaded to the GPU as flat arrays.

## Scope

- CPU-side half-edge construction from triangle index data
- Boundary edge detection (edges adjacent to only one face)
- Non-manifold edge detection and warning
- GPU upload of adjacency arrays for kernel consumption

## Features

| # | Feature | File(s) |
|---|---------|---------|
| 04 | [CPU half-edge construction](feature-04-half-edge-construction.md) | `src/half_edge.cpp` |
| 05 | [GPU adjacency data upload](feature-05-gpu-adjacency-upload.md) | `src/stage2_features.cu` |

## Architecture

```
Triangle indices ──→ build_half_edge_mesh() ──→ HalfEdgeMesh (CPU)
                                                     │
                                                     ├── half_edges[]      (HalfEdge structs)
                                                     ├── edge_to_halfedge[] (uint32 per edge)
                                                     └── edge_to_face[]    (uint32 per edge)
                                                             │
                                                     cudaMemcpy ──→ GPU device arrays
```

## Success Criteria

- [ ] `build_half_edge_mesh()` produces correct adjacency for a cube (12 edges, 24 half-edges)
- [ ] Boundary edges have `opposite == UINT32_MAX`
- [ ] Non-manifold edges are detected and warned about
- [ ] GPU upload produces device arrays ready for kernel consumption

## Dependencies

- Epic 1 (mesh data loaded from OBJ)

## Downstream Dependents

- Epic 4 (Stage 2 feature detection uses the adjacency structure)
