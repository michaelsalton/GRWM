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

// --- Vector math helpers ---

__device__ inline float3 make_sub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float3 cross3(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ inline float length3(float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

// --- Cotangent of angle at vertex b in triangle (a, b, c) ---
// cot(theta) = cos(theta)/sin(theta) = dot(ba,bc) / |cross(ba,bc)|
// Clamped to [-10, 10] per spec Section 3.4
__device__ inline float cotangent(float3 a, float3 b, float3 c) {
    float3 ba = make_sub(a, b);
    float3 bc = make_sub(c, b);

    float d = dot3(ba, bc);
    float cross_len = length3(cross3(ba, bc));

    if (cross_len < 1e-8f) return 10.0f;
    float cot = d / cross_len;
    return fminf(fmaxf(cot, -10.0f), 10.0f);
}

// --- Murmur hash (Murmur3 32-bit finalizer) ---
__device__ inline uint32_t murmur_hash(uint32_t key) {
    key ^= key >> 16;
    key *= 0x85ebca6bu;
    key ^= key >> 13;
    key *= 0xc2b2ae35u;
    key ^= key >> 16;
    return key;
}

// --- Generate slot position from face ID and slot index ---
__device__ inline float2 slot_position(uint32_t face_id, uint32_t slot_index) {
    uint32_t h = murmur_hash(face_id ^ (slot_index * 2654435761u));
    return make_float2(
        float(h & 0xFFFF) / 65535.0f,
        float(h >> 16) / 65535.0f
    );
}

// --- Convert UV to barycentric coordinates ---
// Mapping: (0,0)=v0, (1,0)=v1, (0,1)=v2
__device__ inline float3 uv_to_barycentrics(float2 uv) {
    float w = 1.0f - uv.x - uv.y;
    return make_float3(w, uv.x, uv.y);
}

// --- Barycentric interpolation of scalar values ---
__device__ inline float barycentric_interp(float v0, float v1, float v2, float3 bary) {
    return bary.x * v0 + bary.y * v1 + bary.z * v2;
}

// --- fract (not built into CUDA) ---
__device__ inline float fract(float x) {
    return x - floorf(x);
}

// Wrapper for CUB segmented sort (descending by priority)
void segmented_sort_slots(
    float*    d_keys_in,    float*    d_keys_out,
    uint32_t* d_values_in,  uint32_t* d_values_out,
    uint32_t  total_items,
    uint32_t  num_segments,
    uint32_t* d_offsets);

} // namespace grwm
