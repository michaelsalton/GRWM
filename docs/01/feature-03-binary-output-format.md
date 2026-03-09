# Feature 03: Binary Output Format & I/O

## Context

The pipeline produces three binary files consumed by Gravel at load time. Each file has a 32-byte `PreprocessHeader` followed by raw data. The header format is shared between GRWM and Gravel's loader — both include the same struct definition from `preprocess.h`.

## Requirements

1. Implement `write_curvature_bin()` — header + `float[vertex_count]`
2. Implement `write_features_bin()` — header + `uint8_t[face_count]`
3. Implement `write_slots_bin()` — header + `SlotEntry[face_count × slots_per_face]`
4. All writes use the same `PreprocessHeader` format
5. Create output directory if it doesn't exist
6. Return `false` and print error on write failure

## Files Modified

- `src/visualize.cpp` — implement the three `write_*_bin()` functions

## Implementation Details

### PreprocessHeader (from `preprocess.h`)

```cpp
struct PreprocessHeader {
    uint32_t magic;          // 0x47525650 ("GRVP")
    uint32_t version;        // 1
    uint32_t vertex_count;
    uint32_t face_count;
    uint32_t edge_count;
    uint32_t slots_per_face;
    uint32_t padding[2];     // reserved, set to 0
};
// sizeof(PreprocessHeader) == 32
```

### File Layouts

**curvature.bin:**
```
[PreprocessHeader: 32 bytes]
[float curvature[vertex_count]: vertex_count × 4 bytes]
```

**features.bin:**
```
[PreprocessHeader: 32 bytes]
[uint8_t feature_flag[face_count]: face_count × 1 byte]
```

**slots.bin:**
```
[PreprocessHeader: 32 bytes]
[SlotEntry slots[face_count × slots_per_face]: face_count × slots_per_face × 16 bytes]
```

### Write Function Pattern

```cpp
bool grwm::write_curvature_bin(const std::string& path,
                               const PreprocessHeader& header,
                               const std::vector<float>& curvature) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        fprintf(stderr, "Failed to open %s for writing\n", path.c_str());
        return false;
    }
    out.write(reinterpret_cast<const char*>(&header), sizeof(PreprocessHeader));
    out.write(reinterpret_cast<const char*>(curvature.data()),
              curvature.size() * sizeof(float));
    return out.good();
}
```

The `features_bin` and `slots_bin` functions follow the same pattern with their respective data types.

### Size Estimates

For a 200k face mesh with 64 slots per face:
- `curvature.bin`: 32 + ~400KB = ~400KB (assuming ~100k vertices)
- `features.bin`: 32 + 200KB = ~200KB
- `slots.bin`: 32 + 200,000 × 64 × 16 = ~195MB

## Acceptance Criteria

- [ ] Each output file starts with a valid 32-byte header
- [ ] Header magic is `0x47525650` and version is `1`
- [ ] `curvature.bin` total size = 32 + vertex_count × 4
- [ ] `features.bin` total size = 32 + face_count × 1
- [ ] `slots.bin` total size = 32 + face_count × slots_per_face × 16
- [ ] Files can be read back and deserialized correctly (round-trip test)
- [ ] Write failure returns `false` with error message

## Dependencies

- Feature 01 (build system)
- `PreprocessHeader` and `SlotEntry` structs from `preprocess.h` (already scaffolded)
