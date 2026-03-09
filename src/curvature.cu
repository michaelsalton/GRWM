#include "preprocess.h"
#include "cub_helpers.cuh"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <cstdio>
#include <vector>

namespace grwm {

// --- Feature 07: Cotangent weight assembly ---
// Each face contributes 6 off-diagonal COO entries (3 edges × 2 directions)
// and accumulates diagonal entries via atomicAdd.
// COO layout: face f writes to entries [f*6 .. f*6+5]

__global__ void BuildCotangentWeights(
    const float3* positions,
    const uint3*  faces,
    int*          coo_row,
    int*          coo_col,
    float*        coo_values,
    float*        diagonal,      // [V] diagonal accumulator (negative sum of weights)
    uint32_t      face_count)
{
    uint32_t fid = blockIdx.x * blockDim.x + threadIdx.x;
    if (fid >= face_count) return;

    uint3 f = faces[fid];
    float3 v0 = positions[f.x];
    float3 v1 = positions[f.y];
    float3 v2 = positions[f.z];

    // Cotangent of angle at each vertex (opposite edge gets this weight)
    float cot0 = cotangent(v1, v0, v2);  // angle at v0 → weight for edge (v1,v2)
    float cot1 = cotangent(v2, v1, v0);  // angle at v1 → weight for edge (v0,v2)
    float cot2 = cotangent(v0, v2, v1);  // angle at v2 → weight for edge (v0,v1)

    // Off-diagonal COO entries (6 per face, symmetric pairs)
    uint32_t base = fid * 6;

    // Edge (v0, v1): weight = cot2
    coo_row[base + 0] = f.x;  coo_col[base + 0] = f.y;  coo_values[base + 0] = cot2;
    coo_row[base + 1] = f.y;  coo_col[base + 1] = f.x;  coo_values[base + 1] = cot2;

    // Edge (v1, v2): weight = cot0
    coo_row[base + 2] = f.y;  coo_col[base + 2] = f.z;  coo_values[base + 2] = cot0;
    coo_row[base + 3] = f.z;  coo_col[base + 3] = f.y;  coo_values[base + 3] = cot0;

    // Edge (v2, v0): weight = cot1
    coo_row[base + 4] = f.z;  coo_col[base + 4] = f.x;  coo_values[base + 4] = cot1;
    coo_row[base + 5] = f.x;  coo_col[base + 5] = f.z;  coo_values[base + 5] = cot1;

    // Diagonal: L[i][i] = -sum of weights for vertex i
    atomicAdd(&diagonal[f.x], -(cot2 + cot1));
    atomicAdd(&diagonal[f.y], -(cot0 + cot2));
    atomicAdd(&diagonal[f.z], -(cot1 + cot0));
}

// --- Feature 09: Voronoi area computation ---
// Using 1/3 area approximation (simpler than full mixed Voronoi, sufficient for visualization)

__global__ void ComputeVoronoiAreas(
    const float3* positions,
    const uint3*  faces,
    float*        vertex_areas,
    uint32_t      face_count)
{
    uint32_t fid = blockIdx.x * blockDim.x + threadIdx.x;
    if (fid >= face_count) return;

    uint3 f = faces[fid];
    float3 v0 = positions[f.x];
    float3 v1 = positions[f.y];
    float3 v2 = positions[f.z];

    float3 e1 = make_sub(v1, v0);
    float3 e2 = make_sub(v2, v0);

    float area = 0.5f * length3(cross3(e1, e2));
    float third = area / 3.0f;

    atomicAdd(&vertex_areas[f.x], third);
    atomicAdd(&vertex_areas[f.y], third);
    atomicAdd(&vertex_areas[f.z], third);
}

// --- Curvature magnitude from Laplacian vectors ---

__global__ void ComputeCurvatureMagnitude(
    const float*  laplacian,     // [V*3] Laplacian vectors (Lx,Ly,Lz per vertex)
    const float*  vertex_areas,
    float*        curvature_out,
    uint32_t      vertex_count)
{
    uint32_t vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= vertex_count) return;

    float lx = laplacian[vid * 3 + 0];
    float ly = laplacian[vid * 3 + 1];
    float lz = laplacian[vid * 3 + 2];
    float len = sqrtf(lx * lx + ly * ly + lz * lz);
    float area = vertex_areas[vid];

    curvature_out[vid] = (area > 1e-10f) ? len / (2.0f * area) : 0.0f;
}

// --- Diagonal contribution: laplacian[v] += diagonal[v] * positions[v] ---

__global__ void AddDiagonalContribution(
    float*        laplacian,     // [V*3] in/out
    const float*  diagonal,      // [V]
    const float*  positions,     // [V*3]
    uint32_t      vertex_count)
{
    uint32_t vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= vertex_count) return;

    float d = diagonal[vid];
    laplacian[vid * 3 + 0] += d * positions[vid * 3 + 0];
    laplacian[vid * 3 + 1] += d * positions[vid * 3 + 1];
    laplacian[vid * 3 + 2] += d * positions[vid * 3 + 2];
}

// --- Feature 08 + 10: cuSPARSE SpMV pipeline + host orchestration ---

// Helper macros for cusparse error checking
#define CUSPARSE_CHECK(call)                                                \
    do {                                                                    \
        cusparseStatus_t status = (call);                                   \
        if (status != CUSPARSE_STATUS_SUCCESS) {                            \
            fprintf(stderr, "cuSPARSE error at %s:%d: %d\n",               \
                    __FILE__, __LINE__, (int)status);                       \
            goto cleanup;                                                   \
        }                                                                   \
    } while (0)

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            goto cleanup;                                                   \
        }                                                                   \
    } while (0)

std::vector<float> compute_curvature(const MeshData& mesh) {
    if (mesh.vertex_count == 0 || mesh.face_count == 0) {
        return {};
    }

    uint32_t V = mesh.vertex_count;
    uint32_t F = mesh.face_count;
    uint32_t nnz_offdiag = F * 6;   // 6 off-diagonal COO entries per face

    std::vector<float> result(V, 0.0f);

    // Device pointers
    float3*   d_positions   = nullptr;
    uint3*    d_faces       = nullptr;
    int*      d_coo_row     = nullptr;
    int*      d_coo_col     = nullptr;
    float*    d_coo_val     = nullptr;
    float*    d_diagonal    = nullptr;
    int*      d_csr_offsets = nullptr;
    float*    d_laplacian   = nullptr;  // V*3 (result of SpMM)
    float*    d_positions_f = nullptr;  // V*3 (positions as dense matrix)
    float*    d_areas       = nullptr;
    float*    d_curvature   = nullptr;
    void*     d_workspace   = nullptr;

    cusparseHandle_t     handle    = nullptr;
    cusparseSpMatDescr_t mat_desc  = nullptr;
    cusparseDnMatDescr_t pos_desc  = nullptr;
    cusparseDnMatDescr_t lap_desc  = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_positions,   V * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_faces,       F * sizeof(uint3)));
    CUDA_CHECK(cudaMalloc(&d_coo_row,     nnz_offdiag * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_coo_col,     nnz_offdiag * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_coo_val,     nnz_offdiag * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_diagonal,    V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_csr_offsets, (V + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_laplacian,   V * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_positions_f, V * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_areas,       V * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_curvature,   V * sizeof(float)));

    // Upload mesh data
    CUDA_CHECK(cudaMemcpy(d_positions, mesh.positions.data(),
                          V * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_faces, mesh.indices.data(),
                          F * sizeof(uint3), cudaMemcpyHostToDevice));
    // Positions as flat float array for SpMM (same data, just typed as float*)
    CUDA_CHECK(cudaMemcpy(d_positions_f, mesh.positions.data(),
                          V * 3 * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize accumulators
    CUDA_CHECK(cudaMemset(d_diagonal, 0, V * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_areas,    0, V * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_laplacian, 0, V * 3 * sizeof(float)));

    // --- Step 1: Build cotangent weights (COO + diagonal) ---
    {
        int threads = 256;
        int blocks = (F + threads - 1) / threads;
        BuildCotangentWeights<<<blocks, threads>>>(
            d_positions, d_faces,
            d_coo_row, d_coo_col, d_coo_val,
            d_diagonal, F);
        CUDA_CHECK(cudaGetLastError());
    }

    // --- Step 2: cuSPARSE COO → CSR, then SpMM ---
    CUSPARSE_CHECK(cusparseCreate(&handle));

    // Create COO sparse matrix descriptor (off-diagonal part only)
    CUSPARSE_CHECK(cusparseCreateCoo(
        &mat_desc, V, V, nnz_offdiag,
        d_coo_row, d_coo_col, d_coo_val,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // Convert COO to CSR
    CUSPARSE_CHECK(cusparseXcoo2csr(
        handle, d_coo_row, nnz_offdiag, V,
        d_csr_offsets, CUSPARSE_INDEX_BASE_ZERO));

    // Destroy COO descriptor and create CSR descriptor
    CUSPARSE_CHECK(cusparseDestroySpMat(mat_desc));
    mat_desc = nullptr;

    CUSPARSE_CHECK(cusparseCreateCsr(
        &mat_desc, V, V, nnz_offdiag,
        d_csr_offsets, d_coo_col, d_coo_val,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // Create dense matrix descriptors
    // Positions: V × 3, row-major, leading dimension = 3
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &pos_desc, V, 3, 3,
        d_positions_f, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // Laplacian output: V × 3
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &lap_desc, V, 3, 3,
        d_laplacian, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // SpMM: laplacian = L_offdiag * positions
    {
        float alpha = 1.0f, beta = 0.0f;
        size_t workspace_size = 0;

        CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat_desc, pos_desc, &beta, lap_desc,
            CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &workspace_size));

        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));

        CUSPARSE_CHECK(cusparseSpMM(
            handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat_desc, pos_desc, &beta, lap_desc,
            CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_workspace));
    }

    // --- Step 3: Add diagonal contribution ---
    {
        int threads = 256;
        int blocks = (V + threads - 1) / threads;
        AddDiagonalContribution<<<blocks, threads>>>(
            d_laplacian, d_diagonal, d_positions_f, V);
        CUDA_CHECK(cudaGetLastError());
    }

    // --- Step 4: Compute Voronoi areas ---
    {
        int threads = 256;
        int blocks = (F + threads - 1) / threads;
        ComputeVoronoiAreas<<<blocks, threads>>>(
            d_positions, d_faces, d_areas, F);
        CUDA_CHECK(cudaGetLastError());
    }

    // --- Step 5: Compute curvature magnitude ---
    {
        int threads = 256;
        int blocks = (V + threads - 1) / threads;
        ComputeCurvatureMagnitude<<<blocks, threads>>>(
            d_laplacian, d_areas, d_curvature, V);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(result.data(), d_curvature,
                          V * sizeof(float), cudaMemcpyDeviceToHost));

cleanup:
    if (mat_desc) cusparseDestroySpMat(mat_desc);
    if (pos_desc) cusparseDestroyDnMat(pos_desc);
    if (lap_desc) cusparseDestroyDnMat(lap_desc);
    if (handle)   cusparseDestroy(handle);

    cudaFree(d_positions);
    cudaFree(d_faces);
    cudaFree(d_coo_row);
    cudaFree(d_coo_col);
    cudaFree(d_coo_val);
    cudaFree(d_diagonal);
    cudaFree(d_csr_offsets);
    cudaFree(d_laplacian);
    cudaFree(d_positions_f);
    cudaFree(d_areas);
    cudaFree(d_curvature);
    cudaFree(d_workspace);

    return result;
}

} // namespace grwm
