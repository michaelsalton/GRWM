cmake_minimum_required(VERSION 3.20)
project(GRWM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# --- Dependencies ---
include(FetchContent)
FetchContent_Declare(
    tinyobjloader
    GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
    GIT_TAG        v2.0.0rc13
)
FetchContent_MakeAvailable(tinyobjloader)

find_package(CUDAToolkit REQUIRED)

# --- Core library (linked by main exe and tests) ---
add_library(cuda_preprocess_lib STATIC
    src/mesh_loader.cpp
    src/half_edge.cpp
    src/curvature.cu
    src/features.cu
    src/slots.cu
    src/validate.cu
    src/visualize.cpp
)

target_include_directories(cuda_preprocess_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(cuda_preprocess_lib PUBLIC
    tinyobjloader
    CUDA::cusparse
    CUDA::curand
)

set_target_properties(cuda_preprocess_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "70;75;80;86;89;90"
)

# --- Main executable ---
add_executable(cuda_preprocess src/main.cpp)
target_link_libraries(cuda_preprocess PRIVATE cuda_preprocess_lib)

# --- Tests (optional) ---
option(GRWM_BUILD_TESTS "Build test executables" ON)
if(GRWM_BUILD_TESTS)
    add_subdirectory(tests)
endif()
