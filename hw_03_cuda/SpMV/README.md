## CUDA and CUDA + MPI Implementation of COO SpMV (Sparse Matrix-Vector Multiplication)

### Implementation Details

1. Matrix Storage Format (COO)
The sparse matrix is stored in COO format using three arrays:

- rows[]: Row indices of non-zero elements
- cols[]: Column indices of non-zero elements
- vals[]: Values of non-zero elements

2. Cuda Kernal

```
__global__ void coo_spmv_kernel(int num_nonzeros, const int* rows, const int* cols, 
                               const float* vals, const float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_nonzeros) {
        atomicAdd(&y[rows[idx]], vals[idx] * x[cols[idx]]);
    }
}
```

Key features:

- Each thread processes one non-zero element
- Uses atomicAdd for thread-safe accumulation
  - Each thread could write to the same y address
- Thread index calculation using block and thread IDs

### Discussion
Compared to the sequential COO SpMV calculation, the cuda implementation allows for significantly faster execution, taking advantage of the simple calculation being performed on hundreds of threads at a time. This is ideal for large SpMV calculations, but because of the bottleneck from getting data on/off the GPU, this solution is slower than sequential COO SpMV for small matrices. 

### CUDA + MPI 
The CUDA + MPI implementation is nearly identical to the sequential solution, except each node is given a portion of the nonzero values from the sparse matrix to work with, and the final result is reduced back to the root node. Because the bottle neck for the cuda implementation is getting the data on/off the GPU, the MPI + CUDA implementation is only useful for extremely large datasets. Although since each GPU requires the entire dense X matrix and resulting Y matrix, the total available memory could become an issue for large matrices. This could make the MPI solution more viable since it reduces the size of the sparse matrix that each node would process. 

### Results
can be found in results.txt and the attached images. 