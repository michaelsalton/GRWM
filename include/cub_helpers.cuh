#pragma once

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

namespace grwm {

// CUDA error checking macro
#define GRWM_CUDA_CHECK(call)                                              \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            return;                                                        \
        }                                                                  \
    } while (0)

// Cotangent of angle at vertex b in triangle (a, b, c)
__device__ inline float cotangent(float3 a, float3 b, float3 c) {
    // TODO: implement
    return 0.0f;
}

// Murmur hash for deterministic slot positions
__device__ inline uint32_t murmur_hash(uint32_t key) {
    // TODO: implement
    return key;
}

// Generate slot position from face ID and slot index
__device__ inline float2 slot_position(uint32_t face_id, uint32_t slot_index) {
    // TODO: implement using murmur_hash
    return make_float2(0.0f, 0.0f);
}

// Convert UV to barycentric coordinates
__device__ inline float3 uv_to_barycentrics(float2 uv) {
    // TODO: implement
    return make_float3(1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f);
}

// Barycentric interpolation of scalar values
__device__ inline float barycentric_interp(float v0, float v1, float v2, float3 bary) {
    // TODO: implement
    return 0.0f;
}

// Wrapper for CUB segmented sort (descending by priority)
void segmented_sort_slots(
    float*    d_keys_in,    float*    d_keys_out,
    uint32_t* d_values_in,  uint32_t* d_values_out,
    uint32_t  total_items,
    uint32_t  num_segments,
    uint32_t* d_offsets);

} // namespace grwm
