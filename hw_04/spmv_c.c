#include <stdlib.h>
#include <omp.h>

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

// Function to update an array with (index, value) pairs in bulk
// Parameters:
//   array: pointer to the array to update
//   indices: array of indices
//   values: array of values to add at the corresponding indices
//   count: number of entries to process
void update_array_values(float* array, int* indices, float* values, int count) {
    #pragma omp parallel for
    for (int i = 0; i < count; i++) {
        if (indices[i] >= 0) { // Ensure valid index
            array[indices[i]] += values[i];
        }
    }
}