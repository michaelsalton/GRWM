# Feature 01: CMake + CUDA Build Setup

## Context

GRWM is a mixed C++17/CUDA project. The build system must handle `.cpp` and `.cu` files, link against CUDA toolkit libraries (cuSPARSE, cuRAND), fetch tinyobjloader, and produce both a main executable and a static library for test linking.

## Requirements

1. CMake 3.20+ with `project(GRWM LANGUAGES CXX CUDA)`
2. C++17 and CUDA 17 standards enforced
3. FetchContent for tinyobjloader (pinned to `v2.0.0rc13`)
4. `find_package(CUDAToolkit REQUIRED)` for `CUDA::cusparse` and `CUDA::curand`
5. Static library `cuda_preprocess_lib` containing all source files except `main.cpp`
6. Executable `cuda_preprocess` linked against the static library
7. CUDA separable compilation enabled (for cross-file `__device__` calls)
8. CUDA architectures: 70, 75, 80, 86, 89, 90 (Volta through Hopper)
9. Optional test subdirectory gated by `GRWM_BUILD_TESTS`

## Files Modified

- `CMakeLists.txt` (root) — already scaffolded, verify correctness
- `tests/CMakeLists.txt` — already scaffolded, verify link targets

## Implementation Details

The root CMakeLists.txt structure:

```cmake
cmake_minimum_required(VERSION 3.20)
project(GRWM LANGUAGES CXX CUDA)

# Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Dependencies
include(FetchContent)
FetchContent_Declare(tinyobjloader
    GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
    GIT_TAG        v2.0.0rc13)
FetchContent_MakeAvailable(tinyobjloader)
find_package(CUDAToolkit REQUIRED)

# Static library (all src except main.cpp)
add_library(cuda_preprocess_lib STATIC ...)
target_link_libraries(cuda_preprocess_lib PUBLIC tinyobjloader CUDA::cusparse CUDA::curand)

# Executable
add_executable(cuda_preprocess src/main.cpp)
target_link_libraries(cuda_preprocess PRIVATE cuda_preprocess_lib)
```

### Key Design Notes

- **CUB** ships as a header-only library inside the CUDA toolkit (since CUDA 11). No explicit link target needed — just `#include <cub/cub.cuh>`.
- **Separable compilation** (`CUDA_SEPARABLE_COMPILATION ON`) allows `__device__` functions in `cub_helpers.cuh` to be called from multiple `.cu` translation units.
- **Architecture list** can be overridden: `cmake -B build -DCMAKE_CUDA_ARCHITECTURES=86`

### Windows Considerations

- CUDA toolkit must be installed and `CUDA_PATH` environment variable set
- MSVC version must be compatible with CUDA version (CUDA 12 requires MSVC 2019 or 2022)
- If CMake can't find CUDAToolkit, pass `-DCUDAToolkit_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.x"`

## Acceptance Criteria

- [ ] `cmake -B build` configures without errors (downloads tinyobjloader, finds CUDA)
- [ ] `cmake --build build` compiles all `.cpp` and `.cu` files
- [ ] `cuda_preprocess` executable is produced
- [ ] Test executables (`test_sphere`, `test_cube`, `benchmark`) compile and link
- [ ] Building with `-DGRWM_BUILD_TESTS=OFF` skips test compilation

## Dependencies

None — first feature to implement.
