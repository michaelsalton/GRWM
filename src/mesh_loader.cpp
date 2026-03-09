#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include "preprocess.h"

#include <cstdio>

namespace grwm {

bool load_mesh(const std::string& obj_path, MeshData& out_mesh) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    bool ok = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                               obj_path.c_str());

    if (!warn.empty()) {
        fprintf(stderr, "OBJ warning: %s\n", warn.c_str());
    }
    if (!ok) {
        fprintf(stderr, "OBJ load error: %s\n", err.c_str());
        return false;
    }
    if (attrib.vertices.empty()) {
        fprintf(stderr, "OBJ file contains no vertices\n");
        return false;
    }

    // Copy vertex positions (tinyobjloader stores as flat x,y,z,...)
    out_mesh.positions = attrib.vertices;
    out_mesh.vertex_count = static_cast<uint32_t>(attrib.vertices.size() / 3);

    // Flatten triangle indices from all shapes
    uint32_t skipped_ngons = 0;
    for (const auto& shape : shapes) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            int fv = shape.mesh.num_face_vertices[f];
            if (fv != 3) {
                skipped_ngons++;
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

    if (skipped_ngons > 0) {
        fprintf(stderr, "Warning: skipped %u non-triangle faces\n", skipped_ngons);
    }

    out_mesh.face_count = static_cast<uint32_t>(out_mesh.indices.size() / 3);

    if (out_mesh.face_count == 0) {
        fprintf(stderr, "OBJ file contains no triangle faces\n");
        return false;
    }

    return true;
}

} // namespace grwm
