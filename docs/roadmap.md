# GRWM Implementation Roadmap

## Project Summary

GRWM is a standalone CUDA preprocessing tool that takes a triangle mesh (OBJ) and produces three binary buffer files for the Gravel real-time procedural resurfacing engine:

| Output | Format | Purpose |
|--------|--------|---------|
| `curvature.bin` | `float[V]` | Per-vertex mean curvature for density modulation |
| `features.bin` | `uint8[F]` | Per-face feature edge flags for crease coverage |
| `slots.bin` | `SlotEntry[F × N]` | Per-face priority-sorted element placement grids |

## Feature Roadmap (24 Features)

### Epic 1: Project Foundation
| # | Feature | Files | Status |
|---|---------|-------|--------|
| 01 | CMake + CUDA build setup | `CMakeLists.txt` | Scaffolded |
| 02 | OBJ mesh loading | `src/mesh_loader.cpp` | Stub |
| 03 | Binary output format & I/O | `src/visualize.cpp` | Stub |

### Epic 2: Half-Edge Data Structure
| # | Feature | Files | Status |
|---|---------|-------|--------|
| 04 | CPU half-edge construction | `src/half_edge.cpp` | Stub |
| 05 | GPU adjacency data upload | `src/stage2_features.cu` | Stub |

### Epic 3: Curvature Computation (Stage 1)
| # | Feature | Files | Status |
|---|---------|-------|--------|
| 06 | CUDA device helpers | `include/cub_helpers.cuh` | Stub |
| 07 | Cotangent weight assembly kernel | `src/stage1_curvature.cu` | Stub |
| 08 | cuSPARSE SpMV pipeline | `src/stage1_curvature.cu` | Stub |
| 09 | Voronoi area computation kernel | `src/stage1_curvature.cu` | Stub |
| 10 | Curvature host orchestration | `src/stage1_curvature.cu` | Stub |

### Epic 4: Feature Edge Detection (Stage 2)
| # | Feature | Files | Status |
|---|---------|-------|--------|
| 11 | Face normal computation kernel | `src/stage2_features.cu` | Stub |
| 12 | Dihedral angle detection kernel | `src/stage2_features.cu` | Stub |
| 13 | Feature edge host orchestration | `src/stage2_features.cu` | Stub |

### Epic 5: Slot Priority & Segmented Sort (Stage 3)
| # | Feature | Files | Status |
|---|---------|-------|--------|
| 14 | Slot position hashing | `include/cub_helpers.cuh` | Stub |
| 15 | Priority score computation kernel | `src/stage3_slots.cu` | Stub |
| 16 | CUB segmented sort integration | `src/stage3_slots.cu` | Stub |
| 17 | Slots host orchestration | `src/stage3_slots.cu` | Stub |

### Epic 6: Validation & Visualization
| # | Feature | Files | Status |
|---|---------|-------|--------|
| 18 | Sphere curvature validation | `src/validate.cu` | Stub |
| 19 | Cube feature edge validation | `src/validate.cu` | Stub |
| 20 | Slot sort order validation | `src/validate.cu` | Stub |
| 21 | Visualization output (PLY/OBJ) | `src/visualize.cpp` | Stub |

### Epic 7: Testing & Benchmarking
| # | Feature | Files | Status |
|---|---------|-------|--------|
| 22 | test_sphere executable | `tests/test_sphere.cpp` | Stub |
| 23 | test_cube executable | `tests/test_cube.cpp` | Stub |
| 24 | Benchmark framework | `tests/benchmark.cpp` | Stub |

## Dependency Graph

```
01 ─→ 02 ─→ 03
              │
              ├─→ 04 ─→ 05
              │
              └─→ 06 ─┬─→ 07 ─→ 08 ─→ 09 ─→ 10
                       │
                       └─→ 14 ─→ 15 ─→ 16 ─→ 17
              │
              ├─→ 04 ─→ 05 ─→ 11 ─→ 12 ─→ 13
              │
              ├─→ 10 ─→ 18
              ├─→ 13 ─→ 19
              ├─→ 17 ─→ 20
              │
              ├─→ 10 + 13 ─→ 21
              │
              ├─→ 18 ─→ 22
              ├─→ 19 ─→ 23
              └─→ 10 + 13 + 17 ─→ 24
```

### Parallelizable Groups

These feature groups can be implemented concurrently once their dependencies are met:

| Group | Features | Prerequisite |
|-------|----------|-------------|
| A | 04, 06 | 03 |
| B | 07, 14 | 06 |
| C | 11, 15 | 05, 14 |
| D | 18, 19, 20 | 10, 13, 17 |
| E | 22, 23 | 18, 19 |

## Recommended Implementation Order

**Phase 1 — Foundation** (Features 01-03)
Build system, mesh loading, binary I/O. After this phase, the CLI tool can load an OBJ and write empty output files.

**Phase 2 — Infrastructure** (Features 04-06)
Half-edge structure and CUDA device helpers. These are shared dependencies for all three pipeline stages.

**Phase 3 — Stage 1 Curvature** (Features 07-10)
The most complex stage. Involves sparse matrix assembly, cuSPARSE integration, and multi-kernel orchestration.

**Phase 4 — Stage 2 Features** (Features 11-13)
Simplest stage. Embarrassingly parallel, minimal CUDA complexity.

**Phase 5 — Stage 3 Slots** (Features 14-17)
Moderate complexity. CUB segmented sort is the key technical challenge.

**Phase 6 — Validation & Visualization** (Features 18-21)
Correctness verification and debugging tools. Should be implemented before benchmarking.

**Phase 7 — Testing & Benchmarking** (Features 22-24)
Test executables and performance measurement framework.

## Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language (host) | C++17 | — |
| Language (device) | CUDA | 12+ |
| Build system | CMake | 3.20+ |
| Sparse linear algebra | cuSPARSE | (CUDA toolkit) |
| Parallel algorithms | CUB | (CUDA toolkit) |
| Random numbers | cuRAND | (CUDA toolkit) |
| Mesh loading | tinyobjloader | v2.0.0rc13 |
| GPU target | Compute Capability | 7.0+ (Volta) |
