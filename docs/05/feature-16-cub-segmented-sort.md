# Feature 16: CUB Segmented Sort Integration

## Context

After priorities are computed, the slots within each face must be sorted by descending priority. This is a segmented sort: F independent sort operations, each over N_max elements. CUB's `DeviceSegmentedSort` parallelizes all F sorts across the full GPU, providing 50-100× speedup over sequential CPU sorting.

## Requirements

1. Implement `segmented_sort_slots()` wrapper using CUB
2. Sort by descending priority within each face segment
3. Maintain slot entry association (sort keys carry values)
4. Handle temporary workspace allocation
5. Construct segment offsets array (trivial: 0, N_max, 2×N_max, ...)

## Files Modified

- `src/stage3_slots.cu` — implement `segmented_sort_slots()` body

## Implementation Details

### CUB API

```cpp
#include <cub/device/device_segmented_sort.cuh>

void grwm::segmented_sort_slots(
    float*    d_keys_in,    float*    d_keys_out,
    uint32_t* d_values_in,  uint32_t* d_values_out,
    uint32_t  total_items,
    uint32_t  num_segments,
    uint32_t* d_offsets)
{
    // Step 1: Query temporary storage size
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceSegmentedSort::SortPairsDescending(
        d_temp, temp_bytes,
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        total_items,
        num_segments,
        d_offsets,         // begin offsets
        d_offsets + 1,     // end offsets
        0);                // stream

    // Step 2: Allocate temporary storage
    cudaMalloc(&d_temp, temp_bytes);

    // Step 3: Execute sort
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
```

### Sort Key/Value Mapping

The sort operates on key-value pairs:
- **Keys**: `float priority[F × N_max]` — extracted from SlotEntry.priority
- **Values**: `uint32_t slot_index[F × N_max]` — original slot indices

After sorting, the SlotEntry buffer is reconstructed from the sorted order. Alternatively, use a structure-of-arrays approach during sorting and pack back into SlotEntry afterward.

### Segment Offsets

Since all segments have equal length N_max, offsets are a simple arithmetic sequence:

```cpp
// Host construction
std::vector<uint32_t> offsets(num_segments + 1);
for (uint32_t i = 0; i <= num_segments; ++i)
    offsets[i] = i * slots_per_face;

// Upload to device
uint32_t* d_offsets;
cudaMalloc(&d_offsets, offsets.size() * sizeof(uint32_t));
cudaMemcpy(d_offsets, offsets.data(), ..., cudaMemcpyHostToDevice);
```

### Why Not Sort SlotEntry Directly?

CUB's segmented sort sorts key-value pairs where keys and values are separate arrays. Sorting a struct (SlotEntry) by one field requires either:
1. Extract keys, sort key-value pairs, reconstruct structs (recommended)
2. Custom comparator (not supported by CUB's device-level API)

Approach 1 is used here.

### Performance

For N_max=64 and F=200k:
- Total items: 12.8M
- Segments: 200k
- Items per segment: 64

CUB distributes segments across thread blocks. With 64 items per segment, each segment fits in a single warp's worth of work. The GPU processes thousands of segments simultaneously.

**Expected speedup**: 50-100× vs sequential `std::sort` per face on CPU.

### Temporary Storage

CUB requires a temporary workspace buffer. Size depends on problem parameters and is queried via the first (nullptr) call. Typically a few MB for this problem size.

## Acceptance Criteria

- [ ] After sort, priorities within each face segment are in non-increasing order
- [ ] No slot entries are lost or duplicated during sort
- [ ] Slot positions (u, v) remain correctly associated with their priority
- [ ] Temporary workspace is allocated and freed correctly
- [ ] Sort is deterministic (same input → same output)

## Dependencies

- Feature 15 (priority computation produces unsorted buffer)
