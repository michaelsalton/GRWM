# Feature 14: Slot Position Hashing

## Context

Slot positions must be deterministic — the same (face_id, slot_index) pair must always produce the same 2D position. This is achieved using a Murmur hash function that maps the combined key to a uniformly distributed UV coordinate in [0, 1]².

## Requirements

1. Implement `murmur_hash()` device function (Murmur3 32-bit finalizer)
2. Implement `slot_position()` device function using the hash
3. Positions must be in [0, 1] range for both U and V
4. Distribution must be uniform with no visible patterns
5. Results must be identical across runs and GPU architectures

## Files Modified

- `include/cub_helpers.cuh` — implement `murmur_hash()` and `slot_position()` bodies

## Implementation Details

### Murmur Hash (Murmur3 Finalizer)

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

This is the finalization step of MurmurHash3. It provides excellent avalanche behavior — small input changes produce uniformly distributed output changes.

### Slot Position Generation

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

The constant `2654435761` is the golden ratio × 2³² (Knuth's multiplicative hash constant). XORing with this ensures that sequential slot indices produce well-distributed keys even when face_id is 0.

### UV Distribution

- Lower 16 bits → U coordinate in [0, 1]
- Upper 16 bits → V coordinate in [0, 1]
- Resolution: 65536 distinct values per axis (sufficient for element placement)

## Acceptance Criteria

- [ ] `murmur_hash(0) != 0` and `murmur_hash(1) != 1` (non-trivial mixing)
- [ ] `slot_position(f, s)` returns values in [0, 1] for both components
- [ ] Same inputs always produce same outputs (determinism)
- [ ] Visual inspection: scatter plot of 1000 slot positions shows uniform distribution
- [ ] Adjacent face IDs produce well-distributed positions (no clustering)

## Dependencies

- Feature 06 (part of the same file, must coordinate implementation)
