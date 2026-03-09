#pragma once

#include <cstdint>
#include <vector>

namespace grwm {

struct HalfEdge {
    uint32_t vertex;       // vertex this half-edge points to
    uint32_t face;         // face this half-edge belongs to
    uint32_t opposite;     // index of opposite half-edge (UINT32_MAX if boundary)
    uint32_t next;         // next half-edge in the face loop
};

struct HalfEdgeMesh {
    std::vector<HalfEdge> half_edges;  // 3 per face (he[3f+0], he[3f+1], he[3f+2])
    uint32_t edge_count = 0;           // number of unique undirected edges

    // Flat arrays for GPU upload (derived from half_edges)
    std::vector<uint32_t> he_opposite;    // [3F] opposite half-edge index per half-edge
    std::vector<uint32_t> he_face;        // [3F] face index per half-edge
    std::vector<uint32_t> face_edges;     // [3F] edge index per half-edge (face f uses edges at [3f..3f+2])

    // Per unique-edge data
    std::vector<uint32_t> edge_he;        // [E] one half-edge index per unique edge
};

// Build half-edge structure from triangle index buffer.
HalfEdgeMesh build_half_edge_mesh(
    const std::vector<uint32_t>& indices,
    uint32_t vertex_count,
    uint32_t face_count);

} // namespace grwm
