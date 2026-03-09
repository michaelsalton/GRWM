#include "preprocess.h"
#include "half_edge.h"
#include "cub_helpers.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

namespace grwm {

__global__ void ComputeFaceNormals(
    const float3* positions,
    const uint3*  faces,
    float3*       face_normals,
    uint32_t      face_count)
{
    uint32_t fid = blockIdx.x * blockDim.x + threadIdx.x;
    if (fid >= face_count) return;

    // TODO: compute face normal via cross product of two edge vectors
}

__global__ void DetectFeatureEdges(
    const float3*   face_normals,
    const uint32_t* half_edge_opposite,
    const uint32_t* edge_to_face,
    uint8_t*        edge_flags,
    float           cos_threshold,
    uint32_t        edge_count)
{
    uint32_t eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= edge_count) return;

    // TODO: compare dihedral angle between adjacent face normals
    // against threshold, flag edge if it exceeds
}

__global__ void ReduceEdgeFlagsToFaces(
    const uint8_t*  edge_flags,
    const uint32_t* face_edges, // 3 edge indices per face
    uint8_t*        face_flags,
    uint32_t        face_count)
{
    uint32_t fid = blockIdx.x * blockDim.x + threadIdx.x;
    if (fid >= face_count) return;

    // TODO: set face flag to 1 if any of its 3 edges is flagged
}

std::vector<uint8_t> detect_feature_edges(
    const MeshData& mesh,
    float threshold_degrees)
{
    // TODO: implement feature edge detection pipeline:
    // 1. Build half-edge mesh on CPU
    // 2. Upload adjacency data to device
    // 3. ComputeFaceNormals
    // 4. DetectFeatureEdges
    // 5. ReduceEdgeFlagsToFaces
    // 6. Copy results back to host
    fprintf(stderr, "detect_feature_edges: not yet implemented\n");
    return std::vector<uint8_t>(mesh.face_count, 0);
}

} // namespace grwm
