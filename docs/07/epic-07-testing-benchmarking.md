# Epic 7: Testing & Benchmarking

## Purpose

Standalone test executables for automated validation and a benchmarking framework that measures CUDA vs CPU performance across mesh sizes. These provide reproducible correctness and performance evidence.

## Features

| # | Feature | Focus |
|---|---------|-------|
| 22 | [test_sphere executable](feature-22-test-sphere.md) | Curvature validation on procedural icosphere |
| 23 | [test_cube executable](feature-23-test-cube.md) | Feature edge validation on unit cube |
| 24 | [Benchmark framework](feature-24-benchmark-framework.md) | CUDA event timing, CPU reference comparison |

## Architecture

All test executables link against `cuda_preprocess_lib` (the static library containing all pipeline code except `main.cpp`).

```
tests/
├── CMakeLists.txt       → defines test_sphere, test_cube, benchmark targets
├── test_sphere.cpp      → generates icosphere, runs curvature, checks against 1/R
├── test_cube.cpp        → generates cube, runs feature edges, checks all flagged
└── benchmark.cpp        → loads mesh, times each stage (CUDA events + chrono)
```

## Success Criteria

- [ ] `test_sphere` exits with code 0 on correct implementation
- [ ] `test_cube` exits with code 0 on correct implementation
- [ ] `benchmark` prints timing table for all three stages across mesh sizes
- [ ] All executables compile and run independently of `cuda_preprocess` main exe

## Dependencies

- Epics 3-6 (all pipeline stages and validation must be implemented)
