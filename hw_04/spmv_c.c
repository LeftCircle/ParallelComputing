#include <stdlib.h>

// COO format SpMV implementation
void coo_spmv(int n_rows, int n_cols, int nnz,
              int* rows, int* cols, float* vals,
              float* x, float* y) {
    
    // Initialize result vector to zeros
    for (int i = 0; i < n_rows; i++) {
        y[i] = 0.0f;
    }
    
    // Compute SpMV
    for (int i = 0; i < nnz; i++) {
        y[rows[i]] += vals[i] * x[cols[i]];
    }
}