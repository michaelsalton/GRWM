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
    std::vector<HalfEdge> half_edges;
    uint32_t edge_count = 0;

    // Maps for GPU upload
    std::vector<uint32_t> edge_to_halfedge;
    std::vector<uint32_t> edge_to_face;
};

// Build half-edge structure from triangle index buffer.
HalfEdgeMesh build_half_edge_mesh(
    const std::vector<uint32_t>& indices,
    uint32_t vertex_count,
    uint32_t face_count);

} // namespace grwm
