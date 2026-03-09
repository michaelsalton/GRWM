#include "half_edge.h"

#include <cstdio>

namespace grwm {

HalfEdgeMesh build_half_edge_mesh(
    const std::vector<uint32_t>& indices,
    uint32_t vertex_count,
    uint32_t face_count)
{
    // TODO: build half-edge adjacency structure from triangle indices
    fprintf(stderr, "build_half_edge_mesh: not yet implemented\n");
    return {};
}

} // namespace grwm
