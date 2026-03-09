#include "preprocess.h"
#include "half_edge.h"
#include "cub_helpers.cuh"

#include <cuda_runtime.h>

#include <cmath>
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
    const uint32_t* he_opposite,   // [3F] opposite half-edge index per half-edge
    const uint32_t* he_face,       // [3F] face index per half-edge
    const uint32_t* edge_he_map,   // [E]  one half-edge index per unique edge
    uint8_t*        edge_flags,
    float           cos_threshold,
    uint32_t        edge_count)
{
    uint32_t eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= edge_count) return;

    uint32_t he0 = edge_he_map[eid];
    uint32_t he1 = he_opposite[he0];

    // Boundary edge — not a feature edge
    if (he1 == UINT32_MAX) {
        edge_flags[eid] = 0;
        return;
    }

    uint32_t f0 = he_face[he0];
    uint32_t f1 = he_face[he1];

    float3 n0 = face_normals[f0];
    float3 n1 = face_normals[f1];

    float d = n0.x * n1.x + n0.y * n1.y + n0.z * n1.z;

    // If dot product < cos_threshold, angle > threshold → feature edge
    edge_flags[eid] = (d < cos_threshold) ? 1 : 0;
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
    // Step 1: Build half-edge mesh on CPU
    auto he_mesh = build_half_edge_mesh(mesh.indices, mesh.vertex_count, mesh.face_count);
    if (he_mesh.half_edges.empty()) {
        fprintf(stderr, "detect_feature_edges: half-edge construction failed, returning zeros\n");
        return std::vector<uint8_t>(mesh.face_count, 0);
    }

    uint32_t num_he = mesh.face_count * 3;
    uint32_t edge_count = he_mesh.edge_count;

    // Step 2: Allocate device memory
    float3*   d_positions         = nullptr;
    uint3*    d_faces             = nullptr;
    float3*   d_face_normals      = nullptr;
    uint32_t* d_he_opposite       = nullptr;
    uint32_t* d_he_face           = nullptr;
    uint32_t* d_edge_he_map      = nullptr;
    uint32_t* d_face_edge_indices = nullptr;
    uint8_t*  d_edge_flags        = nullptr;
    uint8_t*  d_face_flags        = nullptr;

    cudaMalloc(&d_positions,         mesh.vertex_count * sizeof(float3));
    cudaMalloc(&d_faces,             mesh.face_count * sizeof(uint3));
    cudaMalloc(&d_face_normals,      mesh.face_count * sizeof(float3));
    cudaMalloc(&d_he_opposite,       num_he * sizeof(uint32_t));
    cudaMalloc(&d_he_face,           num_he * sizeof(uint32_t));
    cudaMalloc(&d_edge_he_map,       edge_count * sizeof(uint32_t));
    cudaMalloc(&d_face_edge_indices, num_he * sizeof(uint32_t));
    cudaMalloc(&d_edge_flags,        edge_count * sizeof(uint8_t));
    cudaMalloc(&d_face_flags,        mesh.face_count * sizeof(uint8_t));

    // Step 3: Upload data to device
    cudaMemcpy(d_positions, mesh.positions.data(),
               mesh.vertex_count * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_faces, mesh.indices.data(),
               mesh.face_count * sizeof(uint3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_he_opposite, he_mesh.he_opposite.data(),
               num_he * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_he_face, he_mesh.he_face.data(),
               num_he * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_he_map, he_mesh.edge_he.data(),
               edge_count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_face_edge_indices, he_mesh.face_edges.data(),
               num_he * sizeof(uint32_t), cudaMemcpyHostToDevice);

    cudaMemset(d_edge_flags, 0, edge_count * sizeof(uint8_t));
    cudaMemset(d_face_flags, 0, mesh.face_count * sizeof(uint8_t));

    // Step 4: Compute face normals
    int threads = 256;
    int face_blocks = (mesh.face_count + threads - 1) / threads;
    ComputeFaceNormals<<<face_blocks, threads>>>(
        d_positions, d_faces, d_face_normals, mesh.face_count);

    // Step 5: Detect feature edges
    float threshold_rad = threshold_degrees * 3.14159265358979f / 180.0f;
    float cos_threshold = cosf(threshold_rad);

    int edge_blocks = (edge_count + threads - 1) / threads;
    DetectFeatureEdges<<<edge_blocks, threads>>>(
        d_face_normals, d_he_opposite, d_he_face, d_edge_he_map,
        d_edge_flags, cos_threshold, edge_count);

    // Step 6: Reduce edge flags to face flags
    ReduceEdgeFlagsToFaces<<<face_blocks, threads>>>(
        d_edge_flags, d_face_edge_indices, d_face_flags, mesh.face_count);

    // Step 7: Copy results back
    std::vector<uint8_t> result(mesh.face_count);
    cudaMemcpy(result.data(), d_face_flags,
               mesh.face_count * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_positions);
    cudaFree(d_faces);
    cudaFree(d_face_normals);
    cudaFree(d_he_opposite);
    cudaFree(d_he_face);
    cudaFree(d_edge_he_map);
    cudaFree(d_face_edge_indices);
    cudaFree(d_edge_flags);
    cudaFree(d_face_flags);

    return result;
}

} // namespace grwm
