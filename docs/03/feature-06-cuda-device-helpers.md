# Feature 06: CUDA Device Helpers

## Context

Multiple CUDA kernels across all three stages share common mathematical utilities. These are implemented as `__device__ inline` functions in `cub_helpers.cuh` and compiled via CUDA separable compilation.

## Requirements

1. `cotangent(a, b, c)` — cotangent of angle at vertex b in triangle (a, b, c)
2. `murmur_hash(key)` — deterministic hash for slot position generation
3. `slot_position(face_id, slot_index)` — 2D position from hash
4. `uv_to_barycentrics(uv)` — convert UV to barycentric coordinates
5. `barycentric_interp(v0, v1, v2, bary)` — interpolate scalar using barycentrics
6. All functions must be numerically stable with clamping where needed

## Files Modified

- `include/cub_helpers.cuh` — implement all `__device__` function bodies

## Implementation Details

### cotangent(a, b, c)

Computes cot(angle at b) using the relationship: cot(θ) = cos(θ)/sin(θ) = dot(ba, bc) / |cross(ba, bc)|

```cuda
__device__ inline float cotangent(float3 a, float3 b, float3 c) {
    float3 ba = make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
    float3 bc = make_float3(c.x - b.x, c.y - b.y, c.z - b.z);

    float dot_val = ba.x * bc.x + ba.y * bc.y + ba.z * bc.z;

    float3 cross_val = make_float3(
        ba.y * bc.z - ba.z * bc.y,
        ba.z * bc.x - ba.x * bc.z,
        ba.x * bc.y - ba.y * bc.x
    );
    float cross_len = sqrtf(cross_val.x * cross_val.x +
                            cross_val.y * cross_val.y +
                            cross_val.z * cross_val.z);

    // Clamp to [-10, 10] to prevent blow-up on degenerate triangles
    if (cross_len < 1e-8f) return 10.0f;
    return fminf(fmaxf(dot_val / cross_len, -10.0f), 10.0f);
}
```

**Numerical note:** Cotangent values blow up as angles approach 0 or π. The clamp range [-10, 10] is specified in the GRWM spec (Section 3.4).

### murmur_hash(key)

Standard Murmur3 finalizer for 32-bit keys:

```cuda
__device__ inline uint32_t murmur_hash(uint32_t key) {
    key ^= key >> 16;
    key *= 0x85ebca6bu;
    key ^= key >> 13;
    key *= 0xc2b2ae35u;
    key ^= key >> 16;
    return key;
}
```

### slot_position(face_id, slot_index)

From the spec (Section 5.2):

```cuda
__device__ inline float2 slot_position(uint32_t face_id, uint32_t slot_index) {
    uint32_t h = murmur_hash(face_id ^ (slot_index * 2654435761u));
    return make_float2(
        float(h & 0xFFFF) / 65535.0f,
        float(h >> 16) / 65535.0f
    );
}
```

### uv_to_barycentrics(uv)

Convert a 2D UV coordinate within a triangle to barycentric coordinates. Using the standard mapping where (0,0) = vertex 0, (1,0) = vertex 1, (0,1) = vertex 2:

```cuda
__device__ inline float3 uv_to_barycentrics(float2 uv) {
    float w = 1.0f - uv.x - uv.y;
    return make_float3(w, uv.x, uv.y);
}
```

### barycentric_interp(v0, v1, v2, bary)

```cuda
__device__ inline float barycentric_interp(float v0, float v1, float v2, float3 bary) {
    return bary.x * v0 + bary.y * v1 + bary.z * v2;
}
```

## Acceptance Criteria

- [ ] `cotangent` returns correct values for known triangles (e.g., equilateral: cot(60°) ≈ 0.577)
- [ ] `cotangent` clamps to [-10, 10] for degenerate triangles
- [ ] `murmur_hash` produces uniform distribution (no visible patterns)
- [ ] `slot_position` is deterministic — same (face_id, slot_index) always produces same position
- [ ] `slot_position` values are in [0, 1] range
- [ ] `barycentric_interp` with bary=(1/3, 1/3, 1/3) returns average of three inputs

## Dependencies

- Feature 01 (CUDA compilation must be working)
