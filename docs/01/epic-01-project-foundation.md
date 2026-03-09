# Epic 1: Project Foundation

## Purpose

Establish the build system, mesh I/O, and binary output format that all subsequent epics depend on. After this epic, the CLI tool can load an OBJ file, process it through stub pipeline stages, and write correctly-formatted (but empty) binary output files.

## Scope

- CMake project with mixed C++17/CUDA compilation
- tinyobjloader integration for OBJ mesh parsing
- Binary file I/O with the PreprocessHeader format shared with Gravel

## Features

| # | Feature | File(s) |
|---|---------|---------|
| 01 | [CMake + CUDA build setup](feature-01-cmake-cuda-setup.md) | `CMakeLists.txt`, `tests/CMakeLists.txt` |
| 02 | [OBJ mesh loading](feature-02-obj-mesh-loading.md) | `src/mesh_loader.cpp` |
| 03 | [Binary output format & I/O](feature-03-binary-output-format.md) | `src/visualize.cpp` |

## Architecture

```
OBJ file ──→ mesh_loader ──→ MeshData struct ──→ [pipeline stages] ──→ binary writers ──→ .bin files
                                                                                         ↑
                                                                              PreprocessHeader (32 bytes)
                                                                              + raw data payload
```

## Success Criteria

- [ ] `cmake -B build && cmake --build build` compiles without errors
- [ ] `./cuda_preprocess mesh.obj --output out/` loads mesh, prints vertex/face counts, writes three .bin files
- [ ] Each .bin file starts with a valid 32-byte PreprocessHeader (magic = 0x47525650)
- [ ] Test executables compile and link against `cuda_preprocess_lib`

## Dependencies

None — this is the foundation epic.

## Downstream Dependents

All other epics depend on Epic 1 for build infrastructure, mesh data, and output I/O.
