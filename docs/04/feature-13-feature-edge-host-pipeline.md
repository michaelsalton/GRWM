# Feature 13: Feature Edge Host Orchestration

## Context

This feature implements the full `detect_feature_edges()` host function, coordinating CPU half-edge construction, GPU memory management, kernel launches, and result copy-back.

## Requirements

1. Build half-edge mesh on CPU (calls `build_half_edge_mesh()`)
2. Allocate device memory for all buffers
3. Upload mesh data and adjacency arrays
4. Launch kernels in sequence: normals → edge detection → face reduction
5. Copy face flags back to host
6. Free all device memory
7. Return `vector<uint8_t>` of length face_count

## Files Modified

- `src/stage2_features.cu` — implement `detect_feature_edges()` body

## Implementation Details

### Pipeline Sequence

```
1. Build half-edge mesh on CPU
   he_mesh = build_half_edge_mesh(mesh.indices, mesh.vertex_count, mesh.face_count)

2. Allocate device memory
   - d_positions:          float3[V]
   - d_faces:              uint3[F]
   - d_face_normals:       float3[F]
   - d_half_edge_opposite: uint32[3F]  (opposite index per half-edge)
   - d_edge_to_face:       uint32[3F]  (face index per half-edge)
   - d_edge_flags:         uint8[E]
   - d_face_edge_indices:  uint32[3F]  (3 edge indices per face)
   - d_face_flags:         uint8[F]

3. Upload data
   - Mesh positions and faces
   - Half-edge opposite array
   - Edge-to-face mapping
   - Face-to-edge-indices mapping

4. Compute threshold
   cos_threshold = cosf(threshold_degrees * PI / 180.0f)

5. Launch ComputeFaceNormals<<<(F+255)/256, 256>>>(...)

6. Launch DetectFeatureEdges<<<(E+255)/256, 256>>>(...)

7. Launch ReduceEdgeFlagsToFaces<<<(F+255)/256, 256>>>(...)

8. Copy d_face_flags to host vector

9. Free all device memory
```

### Memory Usage

For a 200k face mesh:
- Positions: ~1.2 MB (100k vertices × 12 bytes)
- Faces: ~2.4 MB (200k × 12 bytes)
- Normals: ~2.4 MB
- Adjacency: ~4.8 MB
- Flags: ~500 KB
- **Total: ~11 MB** — negligible

### Error Handling

If `build_half_edge_mesh()` fails (returns empty), log a warning and return a zero-filled face flags vector (no feature edges detected). The pipeline continues with degraded quality rather than crashing.

## Acceptance Criteria

- [ ] Returns `vector<uint8_t>` of exactly `face_count` elements
- [ ] Values are 0 or 1 only
- [ ] Cube mesh: all faces flagged (value = 1)
- [ ] Smooth sphere: no faces flagged (all values = 0)
- [ ] No CUDA memory leaks
- [ ] Graceful fallback if half-edge construction fails

## Dependencies

- Feature 04 (half-edge construction)
- Feature 05 (GPU adjacency upload)
- Feature 11 (face normal kernel)
- Feature 12 (detection and reduction kernels)
