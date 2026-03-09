# GRWM Documentation Guide

## Overview

This directory contains comprehensive implementation documentation for **GRWM** (Geometry Resampling and Weighting for Meshes), a CUDA mesh preprocessing pipeline and companion tool to [Gravel](https://github.com/michaelsalton/Gravel).

## Documentation Structure

```
docs/
├── procedure.md                              # This file
├── roadmap.md                                # Master feature roadmap (24 features)
├── GRWM.pdf                                  # Technical specification
│
├── 01/                                       # Epic 1: Project Foundation
│   ├── epic-01-project-foundation.md
│   ├── feature-01-cmake-cuda-setup.md
│   ├── feature-02-obj-mesh-loading.md
│   └── feature-03-binary-output-format.md
│
├── 02/                                       # Epic 2: Half-Edge Data Structure
│   ├── epic-02-half-edge-structure.md
│   ├── feature-04-half-edge-construction.md
│   └── feature-05-gpu-adjacency-upload.md
│
├── 03/                                       # Epic 3: Curvature Computation
│   ├── epic-03-curvature-computation.md
│   ├── feature-06-cuda-device-helpers.md
│   ├── feature-07-cotangent-weight-assembly.md
│   ├── feature-08-cusparse-spmv.md
│   ├── feature-09-voronoi-area.md
│   └── feature-10-curvature-host-pipeline.md
│
├── 04/                                       # Epic 4: Feature Edge Detection
│   ├── epic-04-feature-edge-detection.md
│   ├── feature-11-face-normal-kernel.md
│   ├── feature-12-dihedral-angle-detection.md
│   └── feature-13-feature-edge-host-pipeline.md
│
├── 05/                                       # Epic 5: Slot Priority & Sort
│   ├── epic-05-slot-priority-sort.md
│   ├── feature-14-slot-position-hashing.md
│   ├── feature-15-priority-score-kernel.md
│   ├── feature-16-cub-segmented-sort.md
│   └── feature-17-slots-host-pipeline.md
│
├── 06/                                       # Epic 6: Validation & Visualization
│   ├── epic-06-validation-visualization.md
│   ├── feature-18-sphere-curvature-validation.md
│   ├── feature-19-cube-feature-validation.md
│   ├── feature-20-slot-sort-validation.md
│   └── feature-21-visualization-output.md
│
└── 07/                                       # Epic 7: Testing & Benchmarking
    ├── epic-07-testing-benchmarking.md
    ├── feature-22-test-sphere.md
    ├── feature-23-test-cube.md
    └── feature-24-benchmark-framework.md
```

## How to Read This Documentation

### Roadmap
Start with `roadmap.md` for the full feature list, dependency graph, and implementation order.

### Epics
Each numbered directory contains one **epic** (a major subsystem) with:
- An `epic-NN-*.md` file describing the subsystem's purpose, architecture, and success criteria
- Individual `feature-NN-*.md` files for each implementable unit of work

### Feature Documents
Each feature doc follows a consistent format:
1. **Context** — why this feature exists and what depends on it
2. **Requirements** — what the implementation must do
3. **Files Modified** — which source files are touched
4. **Implementation Details** — pseudocode, kernel signatures, algorithms
5. **Acceptance Criteria** — how to verify correctness
6. **Dependencies** — which features must be completed first

## Implementation Order

Features are numbered 01-24 and should generally be implemented in order, though some can be parallelized. The dependency graph in `roadmap.md` shows which features can run concurrently.

**Critical path**: 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 (curvature pipeline end-to-end)

## Reference

- Full technical specification: `GRWM.pdf`
- Companion project: [Gravel](https://github.com/michaelsalton/Gravel)
- Key paper: Raad et al., "Real-time procedural resurfacing using GPU mesh shader," Eurographics 2025
