# Feature 04: CPU Half-Edge Construction

## Context

The feature edge detection kernel (Stage 2) needs to find the two faces adjacent to each edge. A half-edge data structure provides this in O(1) per edge. Building it on the CPU is fast enough (O(E) time) and avoids the complexity of parallel GPU construction.

## Requirements

1. Construct `HalfEdgeMesh` from flat triangle index array
2. For each directed half-edge, store: target vertex, owning face, opposite half-edge index, next half-edge
3. Pair up opposite half-edges using an edge map `(min_vertex, max_vertex) → half-edge index`
4. Detect boundary edges (`opposite == UINT32_MAX`)
5. Detect and warn about non-manifold edges (more than 2 faces sharing an edge)
6. Populate `edge_to_halfedge` and `edge_to_face` arrays for GPU upload
7. Report `edge_count` (number of unique undirected edges)

## Files Modified

- `src/half_edge.cpp` — implement `build_half_edge_mesh()`

## Implementation Details

### Algorithm

```
For each triangle face f with vertices (v0, v1, v2):
    Create 3 half-edges:
        he[3f+0]: vertex=v1, face=f, next=3f+1
        he[3f+1]: vertex=v2, face=f, next=3f+2
        he[3f+2]: vertex=v0, face=f, next=3f+0

    For each half-edge (vi → vj):
        key = (min(vi,vj), max(vi,vj))
        If key exists in edge_map:
            Link opposite pointers between this half-edge and the stored one
        Else:
            Store this half-edge index in edge_map[key]
```

### Data Structures

```cpp
// Edge map: sorted vertex pair → first half-edge index seen
std::unordered_map<uint64_t, uint32_t> edge_map;

// Pack two uint32 vertex indices into one uint64 key
uint64_t edge_key(uint32_t v0, uint32_t v1) {
    uint32_t lo = std::min(v0, v1);
    uint32_t hi = std::max(v0, v1);
    return (uint64_t(lo) << 32) | uint64_t(hi);
}
```

### Non-Manifold Detection

If an edge key is already paired (both half-edges already have opposites), a third face shares this edge — it's non-manifold. Log a warning with the edge vertices and skip the pairing.

### Output Arrays

After construction:
- `edge_to_halfedge[i]` = index of one half-edge for undirected edge `i`
- `edge_to_face[i]` = face index of half-edge `i / 3` (since each face has exactly 3 half-edges, `face = halfedge_index / 3`)

The `edge_count` is the number of unique keys in `edge_map`.

### Complexity

- Time: O(F) where F = face count (each face processes 3 edges, map operations are O(1) amortized)
- Space: O(E) for the edge map, O(3F) for the half-edge array

### Expected Sizes

| Mesh | Faces | Half-edges | Unique Edges |
|------|-------|------------|--------------|
| Cube | 12 | 36 | 18 |
| Sphere (subdivided) | ~5k | ~15k | ~7.5k |
| Stanford Bunny | ~70k | ~210k | ~105k |

## Acceptance Criteria

- [ ] Cube mesh: 18 unique edges, all interior (no boundary)
- [ ] Open mesh: boundary edges correctly have `opposite == UINT32_MAX`
- [ ] Non-manifold edges produce a warning and are skipped
- [ ] `edge_count` matches `edge_map.size()`
- [ ] Every non-boundary half-edge has `half_edges[he.opposite].opposite == he_index` (symmetric)
- [ ] Every half-edge's `next` chain loops back to itself in exactly 3 steps (triangles only)

## Dependencies

- Feature 02 (mesh data must be loaded to provide indices)
