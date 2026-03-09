# Epic 4: Feature Edge Detection (Stage 2)

## Purpose

Detect sharp creases in the mesh by computing dihedral angles between adjacent face normals. Faces adjacent to feature edges are flagged so Gravel's task shader gives them full element coverage regardless of curvature magnitude. This prevents visual artifacts at silhouettes and creases.

## Pipeline

```
Face indices + Positions ──→ ComputeFaceNormals kernel ──→ float3[F]
                                                               │
Half-edge adjacency (from Epic 2) ──→ DetectFeatureEdges kernel ──→ uint8[E] (edge flags)
                                                                         │
                                      ReduceEdgeFlagsToFaces kernel ──→ uint8[F] (face flags)
```

## Features

| # | Feature | Focus |
|---|---------|-------|
| 11 | [Face normal computation kernel](feature-11-face-normal-kernel.md) | Cross product, one thread per face |
| 12 | [Dihedral angle detection kernel](feature-12-dihedral-angle-detection.md) | Dot product threshold, one thread per edge |
| 13 | [Feature edge host orchestration](feature-13-feature-edge-host-pipeline.md) | `detect_feature_edges()` end-to-end |

## Key Parameters

- `--feature-threshold <deg>`: default 30°. Edges with dihedral angle > threshold are flagged.
- Kernel uses `cos_threshold = cos(threshold_radians)` for efficient comparison (dot product < cos_threshold means angle > threshold)

## Success Criteria

- [ ] Cube mesh: all 12 edges flagged as feature edges (all dihedral angles = 90°)
- [ ] Subdivided sphere: zero feature edges below 30° threshold
- [ ] Boundary edges (adjacent to only one face) are not flagged as feature edges
- [ ] Output is deterministic across runs

## Dependencies

- Epic 1 (mesh data)
- Epic 2 (half-edge adjacency structure)

## Downstream Dependents

- Epic 6 (cube validation test)
