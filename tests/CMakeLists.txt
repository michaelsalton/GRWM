add_executable(test_sphere test_sphere.cpp)
target_link_libraries(test_sphere PRIVATE cuda_preprocess_lib)

add_executable(test_cube test_cube.cpp)
target_link_libraries(test_cube PRIVATE cuda_preprocess_lib)

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PRIVATE cuda_preprocess_lib)
