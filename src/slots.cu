#include "preprocess.h"
#include "cub_helpers.cuh"

#include <cuda_runtime.h>
#include <cub/device/device_segmented_sort.cuh>

#include <cstdio>
#include <vector>

namespace grwm {

// --- Feature 15: Priority score computation ---

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
    uint32_t total = face_count * slots_per_face;
    if (tid >= total) return;

    uint32_t fid = tid / slots_per_face;
    uint32_t sid = tid % slots_per_face;

    uint3 f = faces[fid];

    // Deterministic slot position via Murmur hash
    float2 pos = slot_position(fid, sid);

    // Barycentric coordinates for curvature interpolation
    float3 bary = uv_to_barycentrics(pos);

    // Priority component 1: distance from face center (0.5, 0.5 in UV space)
    float dx = pos.x - 0.5f;
    float dy = pos.y - 0.5f;
    float centerScore = 1.0f - sqrtf(dx * dx + dy * dy) * 1.414f;

    // Priority component 2: interpolated curvature at slot position
    float curvScore = barycentric_interp(
        curvature[f.x], curvature[f.y], curvature[f.z], bary);

    // Priority component 3: jitter (breaks ties, prevents visible banding)
    float jitter = fract(sinf(float(sid) * 127.1f) * 43758.5f);

    slots[tid].u          = pos.x;
    slots[tid].v          = pos.y;
    slots[tid].priority   = w_center * centerScore
                          + w_curv   * curvScore
                          + w_jitter * jitter;
    slots[tid].slot_index = sid;
}

// --- Feature 17 helpers: extract keys/values for sort, reconstruct after ---

__global__ void ExtractSortKeys(
    const SlotEntry* slots,
    float*           keys,
    uint32_t*        values,
    uint32_t         total)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    keys[i]   = slots[i].priority;
    values[i] = i;
}

__global__ void ReconstructSlots(
    const SlotEntry* unsorted_slots,
    const uint32_t*  sorted_indices,
    const float*     sorted_priorities,
    SlotEntry*       sorted_slots,
    uint32_t         total)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    uint32_t src = sorted_indices[i];
    sorted_slots[i].u          = unsorted_slots[src].u;
    sorted_slots[i].v          = unsorted_slots[src].v;
    sorted_slots[i].priority   = sorted_priorities[i];
    sorted_slots[i].slot_index = unsorted_slots[src].slot_index;
}

// --- Feature 16: CUB segmented sort wrapper ---

void segmented_sort_slots(
    float*    d_keys_in,    float*    d_keys_out,
    uint32_t* d_values_in,  uint32_t* d_values_out,
    uint32_t  total_items,
    uint32_t  num_segments,
    uint32_t* d_offsets)
{
    void* d_temp = nullptr;
    size_t temp_bytes = 0;

    // Query workspace size
    cub::DeviceSegmentedSort::SortPairsDescending(
        d_temp, temp_bytes,
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        total_items,
        num_segments,
        d_offsets,
        d_offsets + 1,
        0);

    cudaMalloc(&d_temp, temp_bytes);

    // Execute sort
    cub::DeviceSegmentedSort::SortPairsDescending(
        d_temp, temp_bytes,
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        total_items,
        num_segments,
        d_offsets,
        d_offsets + 1,
        0);

    cudaFree(d_temp);
}

// --- Feature 17: Host orchestration ---

std::vector<SlotEntry> compute_slots(
    const MeshData& mesh,
    const std::vector<float>& curvature,
    uint32_t slots_per_face,
    float w_center, float w_curv, float w_jitter)
{
    uint32_t F = mesh.face_count;
    uint32_t V = mesh.vertex_count;
    uint32_t total = F * slots_per_face;

    if (F == 0 || total == 0) return {};

    // VRAM check
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t required = (size_t)total * (sizeof(SlotEntry) + 4 * sizeof(float) + 4 * sizeof(uint32_t))
                    + F * sizeof(uint3) + V * sizeof(float)
                    + (F + 1) * sizeof(uint32_t);
    if (required > free_mem * 9 / 10) {
        fprintf(stderr, "compute_slots: insufficient VRAM: need %zu MB, have %zu MB free\n",
                required / (1024 * 1024), free_mem / (1024 * 1024));
        return {};
    }

    std::vector<SlotEntry> result(total);

    // Device pointers
    uint3*     d_faces       = nullptr;
    float*     d_curvature   = nullptr;
    SlotEntry* d_slots       = nullptr;   // unsorted
    SlotEntry* d_sorted      = nullptr;   // sorted output
    float*     d_keys_in     = nullptr;
    float*     d_keys_out    = nullptr;
    uint32_t*  d_values_in   = nullptr;
    uint32_t*  d_values_out  = nullptr;
    uint32_t*  d_offsets     = nullptr;

    cudaMalloc(&d_faces,      F * sizeof(uint3));
    cudaMalloc(&d_curvature,  V * sizeof(float));
    cudaMalloc(&d_slots,      total * sizeof(SlotEntry));
    cudaMalloc(&d_sorted,     total * sizeof(SlotEntry));
    cudaMalloc(&d_keys_in,    total * sizeof(float));
    cudaMalloc(&d_keys_out,   total * sizeof(float));
    cudaMalloc(&d_values_in,  total * sizeof(uint32_t));
    cudaMalloc(&d_values_out, total * sizeof(uint32_t));
    cudaMalloc(&d_offsets,    (F + 1) * sizeof(uint32_t));

    // Upload mesh data
    cudaMemcpy(d_faces, mesh.indices.data(),
               F * sizeof(uint3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_curvature, curvature.data(),
               V * sizeof(float), cudaMemcpyHostToDevice);

    // Construct and upload segment offsets: 0, N, 2N, ..., F*N
    std::vector<uint32_t> offsets(F + 1);
    for (uint32_t i = 0; i <= F; ++i)
        offsets[i] = i * slots_per_face;
    cudaMemcpy(d_offsets, offsets.data(),
               (F + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Step 1: Compute slot priorities
    {
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        ComputeSlotPriorities<<<blocks, threads>>>(
            d_faces, d_curvature, d_slots,
            F, slots_per_face,
            w_center, w_curv, w_jitter);
    }

    // Step 2: Extract sort keys and values from SlotEntry buffer
    {
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        ExtractSortKeys<<<blocks, threads>>>(
            d_slots, d_keys_in, d_values_in, total);
    }

    // Step 3: Segmented sort (descending by priority within each face)
    segmented_sort_slots(
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        total, F, d_offsets);

    // Step 4: Reconstruct sorted SlotEntry buffer
    {
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        ReconstructSlots<<<blocks, threads>>>(
            d_slots, d_values_out, d_keys_out, d_sorted, total);
    }

    cudaDeviceSynchronize();

    // Copy sorted results back to host
    cudaMemcpy(result.data(), d_sorted,
               total * sizeof(SlotEntry), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_faces);
    cudaFree(d_curvature);
    cudaFree(d_slots);
    cudaFree(d_sorted);
    cudaFree(d_keys_in);
    cudaFree(d_keys_out);
    cudaFree(d_values_in);
    cudaFree(d_values_out);
    cudaFree(d_offsets);

    return result;
}

} // namespace grwm
