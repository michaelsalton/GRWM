# Feature 21: Visualization Output

## Context

Without Gravel, the pipeline outputs can be visualized using standard 3D viewers. Curvature is mapped to per-vertex colors in a PLY file. Feature edges are written as line segments in an OBJ file.

## Requirements

1. `write_curvature_ply()` — PLY with per-vertex RGB color mapped from curvature
2. `write_feature_edges_obj()` — OBJ with line segments for flagged edges
3. Color mapping: blue (low curvature) → red (high curvature)
4. Files viewable in MeshLab, Blender, or any standard 3D viewer

## Files Modified

- `src/visualize.cpp` — implement visualization functions

## Implementation Details

### Curvature PLY

PLY format with per-vertex color:

```
ply
format ascii 1.0
element vertex <V>
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face <F>
property list uchar int vertex_indices
end_header
<vertex data: x y z r g b>
<face data: 3 i0 i1 i2>
```

Color mapping (blue-to-red heat map):

```cpp
void curvature_to_rgb(float curvature, float min_curv, float max_curv,
                      uint8_t& r, uint8_t& g, uint8_t& b) {
    float t = (curvature - min_curv) / (max_curv - min_curv + 1e-10f);
    t = fmaxf(0.0f, fminf(1.0f, t));

    // Blue → Cyan → Green → Yellow → Red
    if (t < 0.25f) {
        r = 0; g = uint8_t(255 * t * 4); b = 255;
    } else if (t < 0.5f) {
        r = 0; g = 255; b = uint8_t(255 * (1 - (t - 0.25f) * 4));
    } else if (t < 0.75f) {
        r = uint8_t(255 * (t - 0.5f) * 4); g = 255; b = 0;
    } else {
        r = 255; g = uint8_t(255 * (1 - (t - 0.75f) * 4)); b = 0;
    }
}
```

### Feature Edges OBJ

OBJ file with vertices and line segments (`l` entries):

```
# Feature edges visualization
v x0 y0 z0
v x1 y1 z1
...
l 1 2
l 3 4
...
```

Each feature edge contributes two vertices and one line segment. Viewable as wireframe overlay in any 3D viewer.

```cpp
bool write_feature_edges_obj(const std::string& path,
                             const MeshData& mesh,
                             const std::vector<uint8_t>& edge_flags) {
    std::ofstream out(path);
    // For each flagged edge, write its two endpoint vertices
    // and a line element connecting them
    // Edge vertices come from the half-edge structure or index pairs
}
```

### File Sizes

For a 200k face mesh:
- PLY: ~10-20 MB (ASCII format, ~100k vertices)
- Feature edges OBJ: ~1-5 MB (depends on number of feature edges)

## Acceptance Criteria

- [ ] PLY file opens in MeshLab and displays colored mesh
- [ ] High-curvature regions appear red, flat regions appear blue
- [ ] Feature edges OBJ shows wireframe at sharp creases
- [ ] Sphere PLY: uniform color (uniform curvature)
- [ ] Cube feature edges: 12 edge segments visible

## Dependencies

- Feature 10 (curvature data for PLY)
- Feature 13 (feature flags for OBJ edges)
- Feature 04 (half-edge structure for mapping edge flags to vertex pairs)
