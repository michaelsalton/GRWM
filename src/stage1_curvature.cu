#include "preprocess.h"
#include "cub_helpers.cuh"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <cstdio>
#include <vector>

namespace grwm {

__global__ void BuildCotangentWeights(
    const float3* positions,
    const uint3*  faces,
    float*        coo_values,
    uint32_t*     coo_row,
    uint32_t*     coo_col,
    uint32_t      face_count)
{
    uint32_t fid = blockIdx.x * blockDim.x + threadIdx.x;
    if (fid >= face_count) return;

    // TODO: compute cotangent weights for the three edges of this face
    // and atomically accumulate into COO sparse matrix entries
}

__global__ void ComputeVoronoiAreas(
    const float3* positions,
    const uint3*  faces,
    float*        vertex_areas,
    uint32_t      face_count)
{
    uint32_t fid = blockIdx.x * blockDim.x + threadIdx.x;
    if (fid >= face_count) return;

    // TODO: compute mixed Voronoi area contribution from this face
    // and atomically accumulate to each vertex
}

__global__ void ComputeCurvatureMagnitude(
    const float3* laplacian_vectors,
    const float*  vertex_areas,
    float*        curvature_out,
    uint32_t      vertex_count)
{
    uint32_t vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= vertex_count) return;

    // TODO: curvature = |laplacian_vector| / (2 * area)
}

std::vector<float> compute_curvature(const MeshData& mesh) {
    // TODO: implement curvature computation pipeline:
    // 1. Upload mesh data to device
    // 2. BuildCotangentWeights -> COO sparse matrix
    // 3. Convert COO to CSR via cuSPARSE
    // 4. SpMV: L * P via cusparseSpMM -> laplacian vectors
    // 5. ComputeVoronoiAreas
    // 6. ComputeCurvatureMagnitude
    // 7. Copy results back to host
    fprintf(stderr, "compute_curvature: not yet implemented\n");
    return std::vector<float>(mesh.vertex_count, 0.0f);
}

} // namespace grwm
