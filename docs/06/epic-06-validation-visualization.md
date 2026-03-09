# Epic 6: Validation & Visualization

## Purpose

Verify pipeline correctness against analytical ground truth and provide visual debugging tools. Validation ensures curvature, feature edges, and slot sorting produce correct results. Visualization enables inspection without running Gravel.

## Features

| # | Feature | Focus |
|---|---------|-------|
| 18 | [Sphere curvature validation](feature-18-sphere-curvature-validation.md) | H(v) ≈ 1/R for sphere of radius R |
| 19 | [Cube feature edge validation](feature-19-cube-feature-validation.md) | All 12 edges flagged |
| 20 | [Slot sort order validation](feature-20-slot-sort-validation.md) | Priority non-increasing per face |
| 21 | [Visualization output](feature-21-visualization-output.md) | PLY curvature colors, OBJ edge wireframe |

## Invocation

```bash
# Run with validation
./cuda_preprocess mesh.obj --validate

# Run with visualization
./cuda_preprocess mesh.obj --vis-curvature
```

## Success Criteria

- [ ] `--validate` on sphere mesh passes curvature check (MAE < 5% of 1/R)
- [ ] `--validate` on cube mesh passes feature edge check (all edges flagged)
- [ ] `--validate` checks slot sort order (100 random faces sampled)
- [ ] `--vis-curvature` produces viewable PLY file in MeshLab/Blender

## Dependencies

- Epic 3 (curvature computation)
- Epic 4 (feature edge detection)
- Epic 5 (slot computation)
