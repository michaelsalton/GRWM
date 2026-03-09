# Feature 08: cuSPARSE Sparse Matrix-Vector Product

## Context

After the cotangent weight matrix is assembled in COO format, it must be converted to CSR format and multiplied with the vertex position matrix to produce per-vertex Laplacian vectors. cuSPARSE provides optimized routines for both operations.

## Requirements

1. Convert COO sparse matrix to CSR format using cuSPARSE
2. Compute sparse matrix-vector product L × P using `cusparseSpMM`
3. Result is a `float3` per vertex (the mean curvature vector before normalization)
4. Handle cuSPARSE descriptor creation and cleanup
5. Allocate and manage temporary workspace buffer

## Files Modified

- `src/stage1_curvature.cu` — implement SpMV section of `compute_curvature()`

## Implementation Details

### COO to CSR Conversion

```cpp
cusparseHandle_t handle;
cusparseCreate(&handle);

// Create COO descriptor
cusparseSpMatDescr_t coo_desc;
cusparseCreateCoo(&coo_desc,
    vertex_count, vertex_count, nnz,
    d_coo_row, d_coo_col, d_coo_values,
    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

// Convert to CSR
int* d_csr_row_ptr;
cudaMalloc(&d_csr_row_ptr, (vertex_count + 1) * sizeof(int));
cusparseXcoo2csr(handle, d_coo_row, nnz, vertex_count,
                 d_csr_row_ptr, CUSPARSE_INDEX_BASE_ZERO);

// Create CSR descriptor
cusparseSpMatDescr_t csr_desc;
cusparseCreateCsr(&csr_desc,
    vertex_count, vertex_count, nnz,
    d_csr_row_ptr, d_coo_col, d_coo_values,
    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
```

### Sparse Matrix × Dense Matrix (SpMM)

The Laplacian is L × P where P is V × 3 (positions as a dense matrix).

```cpp
// Dense matrix descriptor for positions (V × 3)
cusparseDnMatDescr_t pos_desc, result_desc;
cusparseCreateDnMat(&pos_desc,    vertex_count, 3, 3, d_positions, CUDA_R_32F, CUSPARSE_ORDER_ROW);
cusparseCreateDnMat(&result_desc, vertex_count, 3, 3, d_laplacian, CUDA_R_32F, CUSPARSE_ORDER_ROW);

float alpha = 1.0f, beta = 0.0f;

// Query workspace size
size_t workspace_size;
cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, csr_desc, pos_desc, &beta, result_desc,
    CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &workspace_size);

void* d_workspace;
cudaMalloc(&d_workspace, workspace_size);

// Execute SpMM
cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, csr_desc, pos_desc, &beta, result_desc,
    CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_workspace);
```

### Memory Layout

Positions need to be stored as a V × 3 row-major dense matrix for cuSPARSE. The flat `float` array from `MeshData::positions` (x,y,z,x,y,z,...) is already in this format — just cast to `float*` and set leading dimension = 3.

### Cleanup

```cpp
cusparseDestroySpMat(coo_desc);
cusparseDestroySpMat(csr_desc);
cusparseDestroyDnMat(pos_desc);
cusparseDestroyDnMat(result_desc);
cusparseDestroy(handle);
cudaFree(d_workspace);
cudaFree(d_csr_row_ptr);
```

## Acceptance Criteria

- [ ] COO to CSR conversion produces valid CSR arrays
- [ ] SpMM output has V × 3 float values (Laplacian vectors)
- [ ] For a uniform sphere, all Laplacian vectors point radially inward with equal magnitude
- [ ] No cuSPARSE errors (check all return codes)
- [ ] Workspace buffer is properly allocated and freed

## Dependencies

- Feature 07 (COO matrix must be assembled first)
