#include "preprocess.h"
#include "half_edge.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>

namespace grwm {

// --- Binary output helpers ---

static bool write_bin(const std::string& path,
                      const PreprocessHeader& header,
                      const void* data,
                      size_t data_bytes) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        fprintf(stderr, "Failed to open %s for writing\n", path.c_str());
        return false;
    }
    out.write(reinterpret_cast<const char*>(&header), sizeof(PreprocessHeader));
    if (data_bytes > 0) {
        out.write(reinterpret_cast<const char*>(data), data_bytes);
    }
    if (!out.good()) {
        fprintf(stderr, "Write error on %s\n", path.c_str());
        return false;
    }
    printf("  Wrote %s (%zu bytes)\n", path.c_str(),
           sizeof(PreprocessHeader) + data_bytes);
    return true;
}

bool write_curvature_bin(const std::string& path,
                         const PreprocessHeader& header,
                         const std::vector<float>& curvature)
{
    return write_bin(path, header, curvature.data(),
                     curvature.size() * sizeof(float));
}

bool write_features_bin(const std::string& path,
                        const PreprocessHeader& header,
                        const std::vector<uint8_t>& flags)
{
    return write_bin(path, header, flags.data(),
                     flags.size() * sizeof(uint8_t));
}

bool write_slots_bin(const std::string& path,
                     const PreprocessHeader& header,
                     const std::vector<SlotEntry>& slots)
{
    return write_bin(path, header, slots.data(),
                     slots.size() * sizeof(SlotEntry));
}

// --- Feature 21: Visualization output ---

static void curvature_to_rgb(float curvature, float min_curv, float max_curv,
                              uint8_t& r, uint8_t& g, uint8_t& b) {
    float t = (curvature - min_curv) / (max_curv - min_curv + 1e-10f);
    t = std::max(0.0f, std::min(1.0f, t));

    // Blue -> Cyan -> Green -> Yellow -> Red
    if (t < 0.25f) {
        r = 0;
        g = static_cast<uint8_t>(255.0f * t * 4.0f);
        b = 255;
    } else if (t < 0.5f) {
        r = 0;
        g = 255;
        b = static_cast<uint8_t>(255.0f * (1.0f - (t - 0.25f) * 4.0f));
    } else if (t < 0.75f) {
        r = static_cast<uint8_t>(255.0f * (t - 0.5f) * 4.0f);
        g = 255;
        b = 0;
    } else {
        r = 255;
        g = static_cast<uint8_t>(255.0f * (1.0f - (t - 0.75f) * 4.0f));
        b = 0;
    }
}

bool write_curvature_ply(const std::string& path,
                         const MeshData& mesh,
                         const std::vector<float>& curvature)
{
    if (curvature.size() != mesh.vertex_count) {
        fprintf(stderr, "write_curvature_ply: curvature size mismatch\n");
        return false;
    }

    std::ofstream out(path);
    if (!out.is_open()) {
        fprintf(stderr, "Failed to open %s for writing\n", path.c_str());
        return false;
    }

    // Find curvature range for color mapping
    float min_curv = curvature[0], max_curv = curvature[0];
    for (uint32_t i = 1; i < mesh.vertex_count; ++i) {
        min_curv = std::min(min_curv, curvature[i]);
        max_curv = std::max(max_curv, curvature[i]);
    }

    // PLY header
    out << "ply\n"
        << "format ascii 1.0\n"
        << "element vertex " << mesh.vertex_count << "\n"
        << "property float x\n"
        << "property float y\n"
        << "property float z\n"
        << "property uchar red\n"
        << "property uchar green\n"
        << "property uchar blue\n"
        << "element face " << mesh.face_count << "\n"
        << "property list uchar int vertex_indices\n"
        << "end_header\n";

    // Vertex data with color
    for (uint32_t i = 0; i < mesh.vertex_count; ++i) {
        uint8_t r, g, b;
        curvature_to_rgb(curvature[i], min_curv, max_curv, r, g, b);
        out << mesh.positions[i * 3 + 0] << " "
            << mesh.positions[i * 3 + 1] << " "
            << mesh.positions[i * 3 + 2] << " "
            << (int)r << " " << (int)g << " " << (int)b << "\n";
    }

    // Face data
    for (uint32_t i = 0; i < mesh.face_count; ++i) {
        out << "3 "
            << mesh.indices[i * 3 + 0] << " "
            << mesh.indices[i * 3 + 1] << " "
            << mesh.indices[i * 3 + 2] << "\n";
    }

    if (!out.good()) {
        fprintf(stderr, "Write error on %s\n", path.c_str());
        return false;
    }

    printf("  Wrote %s\n", path.c_str());
    return true;
}

bool write_feature_edges_obj(const std::string& path,
                             const MeshData& mesh,
                             const std::vector<uint8_t>& face_flags)
{
    // Build half-edge mesh to get edge-to-vertex mapping
    auto he_mesh = build_half_edge_mesh(mesh.indices, mesh.vertex_count, mesh.face_count);
    if (he_mesh.half_edges.empty()) {
        fprintf(stderr, "write_feature_edges_obj: half-edge construction failed\n");
        return false;
    }

    std::ofstream out(path);
    if (!out.is_open()) {
        fprintf(stderr, "Failed to open %s for writing\n", path.c_str());
        return false;
    }

    out << "# Feature edges visualization\n";

    // Write all mesh vertices (OBJ indices are 1-based)
    for (uint32_t i = 0; i < mesh.vertex_count; ++i) {
        out << "v "
            << mesh.positions[i * 3 + 0] << " "
            << mesh.positions[i * 3 + 1] << " "
            << mesh.positions[i * 3 + 2] << "\n";
    }

    // For each half-edge, if its face is flagged and the opposite face is also
    // flagged (or boundary), emit the edge as a line segment.
    // Use a set to avoid duplicate edges.
    uint32_t num_he = mesh.face_count * 3;
    uint32_t line_count = 0;

    // Track which unique edges we've already emitted
    std::vector<bool> edge_emitted(he_mesh.edge_count, false);

    for (uint32_t he_idx = 0; he_idx < num_he; ++he_idx) {
        uint32_t eid = he_mesh.face_edges[he_idx];
        if (eid == UINT32_MAX || edge_emitted[eid]) continue;

        uint32_t fid = he_mesh.half_edges[he_idx].face;
        if (!face_flags[fid]) continue;

        // Check opposite face too
        uint32_t opp = he_mesh.half_edges[he_idx].opposite;
        if (opp != UINT32_MAX) {
            uint32_t opp_fid = he_mesh.half_edges[opp].face;
            // Only emit if both adjacent faces are flagged (shared feature edge)
            if (!face_flags[opp_fid]) continue;
        }

        // Get edge vertices from the half-edge
        // Half-edge he_idx goes from some vertex to half_edges[he_idx].vertex
        // The source vertex is half_edges[prev].vertex where prev is the previous he in the face
        uint32_t face_base = (he_idx / 3) * 3;
        uint32_t local = he_idx - face_base;
        uint32_t prev_local = (local + 2) % 3;
        uint32_t prev_he = face_base + prev_local;

        uint32_t v0 = he_mesh.half_edges[prev_he].vertex;
        uint32_t v1 = he_mesh.half_edges[he_idx].vertex;

        // OBJ line (1-based indices)
        out << "l " << (v0 + 1) << " " << (v1 + 1) << "\n";
        edge_emitted[eid] = true;
        line_count++;
    }

    if (!out.good()) {
        fprintf(stderr, "Write error on %s\n", path.c_str());
        return false;
    }

    printf("  Wrote %s (%u edges)\n", path.c_str(), line_count);
    return true;
}

} // namespace grwm
