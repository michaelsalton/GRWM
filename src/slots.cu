#include "preprocess.h"
#include "cub_helpers.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

namespace grwm {

__global__ void ComputeSlotPriorities(
    const uint3*  faces,
    const float*  curvature,
    SlotEntry*    slots,
    uint32_t      face_count,
    uint32_t      slots_per_face,
    float         w_center,
    float         w_curv,
    float         w_jitter)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= face_count * slots_per_face) return;

    // TODO: compute slot position via deterministic hash,
    // evaluate priority score (center + curvature + jitter),
    // write to output buffer
}

std::vector<SlotEntry> compute_slots(
    const MeshData& mesh,
    const std::vector<float>& curvature,
    uint32_t slots_per_face,
    float w_center, float w_curv, float w_jitter)
{
    // TODO: implement slot computation pipeline:
    // 1. Upload mesh data and curvature to device
    // 2. ComputeSlotPriorities kernel
    // 3. Segmented sort via CUB (segmented_sort_slots)
    // 4. Copy results back to host
    fprintf(stderr, "compute_slots: not yet implemented\n");
    return std::vector<SlotEntry>(mesh.face_count * slots_per_face, SlotEntry{});
}

void segmented_sort_slots(
    float*    d_keys_in,    float*    d_keys_out,
    uint32_t* d_values_in,  uint32_t* d_values_out,
    uint32_t  total_items,
    uint32_t  num_segments,
    uint32_t* d_offsets)
{
    // TODO: implement using cub::DeviceSegmentedSort::SortPairsDescending
    fprintf(stderr, "segmented_sort_slots: not yet implemented\n");
}

} // namespace grwm
