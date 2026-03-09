# Feature 17: Slots Host Orchestration

## Context

This feature ties together slot position generation, priority computation, and segmented sorting into the `compute_slots()` host function. It manages device memory, kernel launches, and result copy-back.

## Requirements

1. Implement `compute_slots()` end-to-end
2. Manage device memory for slot buffer, curvature, faces, offsets, sort workspace
3. Launch `ComputeSlotPriorities` kernel
4. Extract keys/values, run segmented sort, reconstruct SlotEntry array
5. Copy sorted results back to host
6. Free all device memory

## Files Modified

- `src/stage3_slots.cu` — implement `compute_slots()` body

## Implementation Details

### Pipeline Sequence

```
1. Allocate device memory
   - d_faces:        uint3[F]
   - d_curvature:    float[V]
   - d_slots:        SlotEntry[F × N_max]
   - d_keys_in:      float[F × N_max]     (priorities for sort)
   - d_keys_out:     float[F × N_max]
   - d_values_in:    uint32[F × N_max]    (slot indices for sort)
   - d_values_out:   uint32[F × N_max]
   - d_offsets:      uint32[F + 1]

2. Upload mesh faces and curvature to device

3. Construct and upload segment offsets (0, N, 2N, ..., F×N)

4. Launch ComputeSlotPriorities<<<(F*N+255)/256, 256>>>(...)
   → Populates d_slots

5. Extract sort keys (priorities) and values (slot indices) from d_slots
   → Use a simple extraction kernel or cudaMemcpy2D

6. Call segmented_sort_slots(d_keys_in, d_keys_out, d_values_in, d_values_out, ...)

7. Reconstruct sorted SlotEntry buffer from sorted keys/values
   → Use a reconstruction kernel that reads sorted priority + index
     and looks up original (u, v) from the unsorted buffer

8. Copy d_slots (sorted) back to host

9. Free all device memory
```

### Extraction Kernel

```cuda
__global__ void ExtractSortKeys(
    const SlotEntry* slots,
    float*           keys,
    uint32_t*        values,
    uint32_t         total)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    keys[i]   = slots[i].priority;
    values[i] = i;  // use global index as value to reconstruct later
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
```

### Memory Usage

For F=200k, N_max=64:
- Slot buffer: 200k × 64 × 16 = 195 MB
- Sort keys (×2): 200k × 64 × 4 × 2 = 98 MB
- Sort values (×2): 200k × 64 × 4 × 2 = 98 MB
- Offsets: 800 KB
- CUB workspace: ~few MB
- **Total: ~395 MB**

For 1M faces: ~2 GB total. The spec recommends `--slots 32` for GPUs with <4GB VRAM.

### VRAM Check

Before allocation, query available device memory:
```cpp
size_t free_mem, total_mem;
cudaMemGetInfo(&free_mem, &total_mem);
size_t required = face_count * slots_per_face * (16 + 4*4 + 4*4) + ...;
if (required > free_mem * 0.9) {
    fprintf(stderr, "Insufficient VRAM: need %zu MB, have %zu MB free\n",
            required / (1024*1024), free_mem / (1024*1024));
    return {};
}
```

## Acceptance Criteria

- [ ] Returns `vector<SlotEntry>` of exactly `face_count × slots_per_face` elements
- [ ] Priorities within each face segment are in non-increasing order
- [ ] UV coordinates match deterministic hash output
- [ ] No CUDA memory leaks
- [ ] Graceful error message if insufficient VRAM
- [ ] Output is deterministic across runs

## Dependencies

- Feature 14 (slot position hashing)
- Feature 15 (priority kernel)
- Feature 16 (CUB segmented sort)
- Feature 10 (curvature buffer as input)
