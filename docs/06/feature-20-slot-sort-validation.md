# Feature 20: Slot Sort Order Validation

## Context

After segmented sorting, priorities within each face must be in non-increasing order. Additionally, slot positions must match the deterministic hash function output. This validation samples random faces and verifies both invariants.

## Requirements

1. Sample 100 random faces (or all faces if < 100)
2. For each sampled face, verify priorities are in non-increasing order
3. Verify slot positions match `slot_position(face_id, slot_index)` output
4. Report number of violations found

## Files Modified

- `src/validate.cu` — implement `validate_slots()` body

## Implementation Details

### Sort Order Check

```cpp
bool validate_slots(const std::vector<SlotEntry>& slots,
                    uint32_t face_count, uint32_t slots_per_face) {
    uint32_t violations = 0;
    uint32_t samples = std::min(face_count, 100u);

    // Deterministic sampling (every N-th face)
    uint32_t step = face_count / samples;

    for (uint32_t s = 0; s < samples; ++s) {
        uint32_t fid = s * step;
        uint32_t base = fid * slots_per_face;

        // Check sort order
        for (uint32_t i = 1; i < slots_per_face; ++i) {
            if (slots[base + i].priority > slots[base + i - 1].priority) {
                violations++;
                if (violations <= 5) {
                    printf("Sort violation: face %u, slot %u: priority %.4f > %.4f\n",
                           fid, i, slots[base+i].priority, slots[base+i-1].priority);
                }
            }
        }
    }

    printf("Slot validation: sampled %u faces, found %u sort violations\n",
           samples, violations);
    return violations == 0;
}
```

### Position Hash Check

For each sampled face, verify that the slot positions in the output match what `slot_position()` would produce. This requires a CPU reimplementation of the Murmur hash:

```cpp
uint32_t cpu_murmur_hash(uint32_t key) {
    key ^= key >> 16;
    key *= 0x85ebca6bu;
    key ^= key >> 13;
    key *= 0xc2b2ae35u;
    key ^= key >> 16;
    return key;
}

std::pair<float, float> cpu_slot_position(uint32_t face_id, uint32_t slot_index) {
    uint32_t h = cpu_murmur_hash(face_id ^ (slot_index * 2654435761u));
    return {float(h & 0xFFFF) / 65535.0f, float(h >> 16) / 65535.0f};
}
```

## Acceptance Criteria

- [ ] Zero sort order violations across all sampled faces
- [ ] Slot positions match hash function output (within floating-point tolerance)
- [ ] Validation completes quickly (< 100ms for 200k face mesh)
- [ ] Clear reporting of any violations found

## Dependencies

- Feature 17 (slot computation must be implemented)
