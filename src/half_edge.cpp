#include "half_edge.h"

#include <algorithm>
#include <cstdio>
#include <unordered_map>

namespace grwm {

// Pack two vertex indices into a single uint64 key (sorted)
static uint64_t edge_key(uint32_t v0, uint32_t v1) {
    uint32_t lo = std::min(v0, v1);
    uint32_t hi = std::max(v0, v1);
    return (uint64_t(lo) << 32) | uint64_t(hi);
}

HalfEdgeMesh build_half_edge_mesh(
    const std::vector<uint32_t>& indices,
    uint32_t vertex_count,
    uint32_t face_count)
{
    if (indices.size() < 3 || face_count == 0) {
        fprintf(stderr, "build_half_edge_mesh: empty input\n");
        return {};
    }

    HalfEdgeMesh mesh;
    uint32_t num_he = face_count * 3;
    mesh.half_edges.resize(num_he);

    // Phase 1: Create half-edges for each face
    // Half-edges for face f are at indices 3f+0, 3f+1, 3f+2
    // Edge ordering per face: (v0→v1), (v1→v2), (v2→v0)
    for (uint32_t f = 0; f < face_count; ++f) {
        uint32_t v0 = indices[f * 3 + 0];
        uint32_t v1 = indices[f * 3 + 1];
        uint32_t v2 = indices[f * 3 + 2];

        uint32_t base = f * 3;

        // he[base+0]: v0 → v1
        mesh.half_edges[base + 0].vertex   = v1;
        mesh.half_edges[base + 0].face     = f;
        mesh.half_edges[base + 0].opposite = UINT32_MAX;
        mesh.half_edges[base + 0].next     = base + 1;

        // he[base+1]: v1 → v2
        mesh.half_edges[base + 1].vertex   = v2;
        mesh.half_edges[base + 1].face     = f;
        mesh.half_edges[base + 1].opposite = UINT32_MAX;
        mesh.half_edges[base + 1].next     = base + 2;

        // he[base+2]: v2 → v0
        mesh.half_edges[base + 2].vertex   = v0;
        mesh.half_edges[base + 2].face     = f;
        mesh.half_edges[base + 2].opposite = UINT32_MAX;
        mesh.half_edges[base + 2].next     = base + 0;
    }

    // Phase 2: Pair opposite half-edges using an edge map
    // Key: sorted (lo, hi) vertex pair → first half-edge index seen
    std::unordered_map<uint64_t, uint32_t> edge_map;
    edge_map.reserve(num_he);

    // Also assign unique edge indices
    // face_edges[he_idx] = unique edge index for this half-edge's undirected edge
    mesh.face_edges.resize(num_he, UINT32_MAX);
    uint32_t next_edge_id = 0;
    uint32_t non_manifold_count = 0;

    for (uint32_t f = 0; f < face_count; ++f) {
        uint32_t v[3] = {
            indices[f * 3 + 0],
            indices[f * 3 + 1],
            indices[f * 3 + 2]
        };

        // Edges: (v[0],v[1]) at he 3f+0, (v[1],v[2]) at he 3f+1, (v[2],v[0]) at he 3f+2
        uint32_t from[3] = { v[0], v[1], v[2] };
        uint32_t to[3]   = { v[1], v[2], v[0] };

        for (int e = 0; e < 3; ++e) {
            uint32_t he_idx = f * 3 + e;
            uint64_t key = edge_key(from[e], to[e]);

            auto it = edge_map.find(key);
            if (it == edge_map.end()) {
                // First time seeing this edge — store this half-edge
                edge_map[key] = he_idx;
                // Assign a new unique edge index
                mesh.face_edges[he_idx] = next_edge_id;
                next_edge_id++;
            } else {
                uint32_t other = it->second;
                if (mesh.half_edges[other].opposite != UINT32_MAX) {
                    // Already paired — non-manifold edge (3+ faces sharing an edge)
                    non_manifold_count++;
                    mesh.face_edges[he_idx] = mesh.face_edges[other]; // same edge id
                    continue;
                }
                // Pair the two half-edges
                mesh.half_edges[he_idx].opposite = other;
                mesh.half_edges[other].opposite  = he_idx;
                // Same unique edge index as the paired half-edge
                mesh.face_edges[he_idx] = mesh.face_edges[other];
            }
        }
    }

    mesh.edge_count = next_edge_id;

    if (non_manifold_count > 0) {
        fprintf(stderr, "Warning: %u non-manifold edges detected\n", non_manifold_count);
    }

    // Count boundary edges
    uint32_t boundary_count = 0;
    for (const auto& he : mesh.half_edges) {
        if (he.opposite == UINT32_MAX) boundary_count++;
    }
    if (boundary_count > 0) {
        printf("  Half-edge mesh: %u edges, %u boundary half-edges\n",
               mesh.edge_count, boundary_count);
    } else {
        printf("  Half-edge mesh: %u edges (closed manifold)\n", mesh.edge_count);
    }

    // Phase 3: Build flat GPU-upload arrays
    mesh.he_opposite.resize(num_he);
    mesh.he_face.resize(num_he);
    for (uint32_t i = 0; i < num_he; ++i) {
        mesh.he_opposite[i] = mesh.half_edges[i].opposite;
        mesh.he_face[i]     = mesh.half_edges[i].face;
    }

    // Build per-unique-edge → one half-edge mapping
    mesh.edge_he.resize(mesh.edge_count, UINT32_MAX);
    for (uint32_t i = 0; i < num_he; ++i) {
        uint32_t eid = mesh.face_edges[i];
        if (eid < mesh.edge_count && mesh.edge_he[eid] == UINT32_MAX) {
            mesh.edge_he[eid] = i;
        }
    }

    return mesh;
}

} // namespace grwm
