# Feature 15: Priority Score Computation Kernel

## Context

Each slot's priority determines which elements appear first as LOD increases. The priority is a weighted combination of three factors: distance from face center, local curvature, and a small jitter term. Higher-priority slots are activated first at low LOD levels.

## Requirements

1. One CUDA thread per slot (total threads = face_count × slots_per_face)
2. Compute face ID and slot index from thread ID
3. Generate slot position using `slot_position()` hash
4. Compute three priority components: center, curvature, jitter
5. Write SlotEntry (u, v, priority, slot_index) to output buffer
6. Tunable weights via kernel parameters (w_center, w_curv, w_jitter)

## Files Modified

- `src/stage3_slots.cu` — implement `ComputeSlotPriorities` kernel body

## Implementation Details

### Kernel

```cuda
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

    uint32_t fid = tid / slots_per_face;
    uint32_t sid = tid % slots_per_face;

    uint3 f = faces[fid];

    // Deterministic slot position
    float2 pos = slot_position(fid, sid);

    // Barycentric coordinates for curvature interpolation
    float3 bary = uv_to_barycentrics(pos);

    // Priority component 1: distance from face center
    // Center is at (0.5, 0.5) in UV space
    float dx = pos.x - 0.5f;
    float dy = pos.y - 0.5f;
    float centerScore = 1.0f - sqrtf(dx*dx + dy*dy) * 1.414f;

    // Priority component 2: curvature at slot position
    float curvScore = barycentric_interp(
        curvature[f.x], curvature[f.y], curvature[f.z], bary);

    // Priority component 3: jitter (breaks ties, prevents patterns)
    float jitter = fract(sinf(float(sid) * 127.1f) * 43758.5f);

    // Weighted combination
    slots[tid].u          = pos.x;
    slots[tid].v          = pos.y;
    slots[tid].priority   = w_center * centerScore
                          + w_curv   * curvScore
                          + w_jitter * jitter;
    slots[tid].slot_index = sid;
}
```

### Priority Component Details

| Component | Weight | Range | Purpose |
|-----------|--------|-------|---------|
| centerScore | 0.5 | [~-0.2, 1.0] | Prefer centrally placed elements at low LOD |
| curvScore | 0.4 | [0, max_curv] | Concentrate density at high-curvature regions |
| jitter | 0.1 | [0, 1) | Break ties, prevent visible priority banding |

### fract() Helper

CUDA doesn't have a built-in `fract()`. Implement as:
```cuda
__device__ inline float fract(float x) {
    return x - floorf(x);
}
```

### Thread Count

For 200k faces × 64 slots = 12.8M threads. Block size 256 → 50,000 blocks. Well within GPU limits.

## Acceptance Criteria

- [ ] Output buffer has face_count × slots_per_face entries
- [ ] All slot_index values match their position in the per-face segment (sid = tid % N_max)
- [ ] UV coordinates match `slot_position()` output
- [ ] Center slots (near 0.5, 0.5) have higher priority than edge slots
- [ ] High-curvature vertices contribute higher curvScore
- [ ] Changing weights changes the priority ordering

## Dependencies

- Feature 14 (slot_position and murmur_hash)
- Feature 06 (uv_to_barycentrics, barycentric_interp)
- Curvature buffer from Stage 1 (Feature 10)
