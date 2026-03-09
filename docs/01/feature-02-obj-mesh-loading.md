# Feature 02: OBJ Mesh Loading

## Context

All pipeline stages operate on mesh data (vertex positions and triangle indices). The mesh loader parses standard Wavefront OBJ files using tinyobjloader and populates the `MeshData` struct that the rest of the pipeline consumes.

## Requirements

1. Parse `.obj` files using tinyobjloader
2. Extract vertex positions into a flat `float` array (`x,y,z,x,y,z,...`)
3. Extract triangle indices into a flat `uint32_t` array (`i0,i1,i2,i0,i1,i2,...`)
4. Handle triangulated meshes (reject or triangulate n-gons)
5. Set `vertex_count` and `face_count` in `MeshData`
6. Report errors clearly (file not found, parse errors, non-manifold warnings)

## Files Modified

- `src/mesh_loader.cpp` — implement `load_mesh()`

## Implementation Details

```cpp
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

bool grwm::load_mesh(const std::string& obj_path, MeshData& out_mesh) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    bool ok = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, obj_path.c_str());

    if (!ok) {
        fprintf(stderr, "OBJ load error: %s\n", err.c_str());
        return false;
    }

    // Copy positions directly (tinyobjloader stores as flat float array)
    out_mesh.positions = attrib.vertices;  // already flat x,y,z,...
    out_mesh.vertex_count = static_cast<uint32_t>(attrib.vertices.size() / 3);

    // Flatten triangle indices from all shapes
    for (const auto& shape : shapes) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            int fv = shape.mesh.num_face_vertices[f];
            if (fv != 3) {
                fprintf(stderr, "Warning: non-triangle face (vertices=%d), skipping\n", fv);
                index_offset += fv;
                continue;
            }
            for (int v = 0; v < 3; ++v) {
                out_mesh.indices.push_back(
                    static_cast<uint32_t>(shape.mesh.indices[index_offset + v].vertex_index));
            }
            index_offset += fv;
        }
    }

    out_mesh.face_count = static_cast<uint32_t>(out_mesh.indices.size() / 3);
    return true;
}
```

### Key Notes

- `#define TINYOBJLOADER_IMPLEMENTATION` must appear in exactly one `.cpp` file (this one)
- tinyobjloader's `attrib.vertices` is already a flat `std::vector<float>` in `x,y,z` order
- We only need position data — normals and texcoords from the OBJ are ignored (normals are computed by the pipeline, UVs are not used)
- Non-triangle faces are skipped with a warning; for production use, a triangulation step could be added

## Acceptance Criteria

- [ ] `load_mesh("path/to/mesh.obj", mesh)` returns `true` and populates `MeshData`
- [ ] `mesh.vertex_count` matches expected vertex count
- [ ] `mesh.face_count` matches expected face count
- [ ] `mesh.positions.size() == vertex_count * 3`
- [ ] `mesh.indices.size() == face_count * 3`
- [ ] Loading a nonexistent file returns `false` with an error message
- [ ] Loading a mesh with n-gons prints a warning and only includes triangulated faces

## Dependencies

- Feature 01 (build system must compile and link tinyobjloader)
