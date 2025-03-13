#include "utils.h"


void matmul(float* A, float* B, float* C, int m, int n, int k) {
    // Initialize output matrix to zero
    memset(C, 0, m * n * sizeof(float));
	// C[i,j] = sum(A[i,p] * B[p,j])
	for (int i = 0; i < m; i++) {
		for (int p = 0; p < k; p++) {
			for (int j = 0; j < n; j++) {
				C[i * n + j] += A[i * k + p] * B[p * n + j];
			}
		}
	}
}
//     // C[i,j] = sum(A[i,p] * B[p,j])
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             float sum = 0.0f;
//             for (int p = 0; p < k; p++) {
//                 // A is m×k, B is k×n
//                 float a_val = A[i * k + p];       // A[i,p]
//                 float b_val = B[p * n + j];       // B[p,j]
//                 sum += a_val * b_val;
//             }
//             C[i * n + j] = sum;
//         }
//     }
// }

void verify_result(float* C_global, float* A, float* B, int m, int n, int k) {
    int errors = 0;
    float tolerance = 1e-5;
    // Perform reference matrix multiplication
    float* C_ref = (float*)calloc(m * n, sizeof(float));
    if (!C_ref) {
        printf("Error: Failed to allocate memory for C_ref\n");
        return;
    }

    // Compute reference result
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A[i*k + p] * B[p*n + j];
            }
            C_ref[i*n + j] = sum;
        }
    }

    // Compute detailed error statistics
    float max_error = 0.0f;
    float avg_error = 0.0f;
    int max_error_index = -1;
    int max_error_i = -1, max_error_j = -1;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i*n + j;
            float curr_error = fabs(C_global[idx] - C_ref[idx]);
            avg_error += curr_error;
            
            if (curr_error > max_error) {
                max_error = curr_error;
                max_error_index = idx;
                max_error_i = i;
                max_error_j = j;
            }
            
            if (curr_error > tolerance) {
                errors++;                if (errors <= 5) {
                    printf("Error at position [%d,%d]: C_global=%.6f, C_ref=%.6f, diff=%.6f\n",
                           i, j, C_global[idx], C_ref[idx], curr_error);
                }
            }
        }
    }
    avg_error /= (m * n);

    // Print detailed verification results
    printf("\nVerification Results:\n");
    printf("  Total Elements: %d\n", m * n);
    printf("  Errors: %d (%.2f%%)\n", errors, 100.0f * errors / (m * n));
    printf("  Max Error: %e at position [%d,%d] (index %d)\n", 
           max_error, max_error_i, max_error_j, max_error_index);
    printf("  Average Error: %e\n", avg_error);

    free(C_ref);
}

// Utility function to load matrix from file or generate
float* generate_matrix_A(int rows, int cols, int rank) {
    float* matrix = malloc(rows * cols * sizeof(float));
    srand(42);
    
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
    
    return matrix;
}


float* generate_matrix_B(int rows, int cols, int rank) {
    float* matrix = malloc(rows * cols * sizeof(float));
    srand(142);

    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }

    return matrix;
}

float* generate_int_matrix(int rows, int cols, int rank) {
	float* matrix = malloc(rows * cols * sizeof(float));
	
	for (int i = 0; i < rows * cols; i++) {
		matrix[i] = i;
	}
	
	return matrix;
}


// /**
//  * Generates a zero-initialized matrix of given dimensions
//  * @param rows Number of rows
//  * @param cols Number of columns
//  * @return Pointer to allocated matrix (caller must free)
//  */
// float* generate_matrix(int rows, int cols){
// 	float* matrix = (float*)calloc(rows*cols, sizeof(float));
// 	return matrix;
// }

// Matrix* init_c_matrix_for_stationary_c(int m, int k, int n, int grid_size, int rank){
// 	Matrix *local_c = (Matrix*)malloc(sizeof(Matrix));
// 	local_c->rows = m / grid_size;
// 	local_c->cols = n / grid_size;
// 	local_c->matrix = (float*)calloc(local_c->rows * local_c->cols, sizeof(float));
// 	return local_c;
// }

void init_a_matrix_for_stationary_c_summa(float* A, int m, int k,int n_processors, int rank){
	int grid_size = (int)sqrt(n_processors);
	int block_m = ceil(m / grid_size);
	int block_k = ceil(k / grid_size);
	//A->rows = block_m;
	//A->cols = block_k;
	//A->matrix = (float*)calloc(A->rows * A->cols, sizeof(float));

}

void scatter_row_major_matrix(float* global_matrix, float* local_matrix, int m, int k,
						      int grid_size, int rank, int size, MPI_Comm comm) {
	// After local_a allocation, replace the commented scattering code with:
	int *sendcounts = (int*)malloc(size * sizeof(int));
	int *displs = (int*)malloc(size * sizeof(int));
	
	int local_a_rows = m / grid_size;
	int local_a_cols = k / grid_size;
	int blocksize = local_a_rows * local_a_cols;  // Size of each block

	int coords[2];

	set_send_offset_for_block_scat_gath(sendcounts, displs, m, k, grid_size, comm);

	if (rank == 0){
		for (int i = 0; i < size; i++){
			printf("Sendcounts = %d\n", sendcounts[i]);
			printf("Displacement = %d\n", displs[i]);
		}
	}
	MPI_Datatype blocktype = create_block_type(m, k, grid_size);

	// Scatter the blocks
	MPI_Scatterv(global_matrix, sendcounts, displs, blocktype,
		local_matrix, blocksize, MPI_FLOAT,
		0, comm);

	MPI_Type_free(&blocktype);
	free(sendcounts);
	free(displs);
}

MPI_Datatype create_block_type(int m, int k, int grid_size) {
    MPI_Datatype tmp_type, block_type;
    
    // Create vector type for one row of the block
    MPI_Type_vector(m/grid_size,    // number of blocks (rows)
                    k/grid_size,    // elements per block (columns)
                    k,              // stride between blocks (full matrix width)
                    MPI_FLOAT,      // element datatype
                    &tmp_type);     // new datatype
    
    // Create resized type to handle displacements correctly
    MPI_Type_create_resized(tmp_type, 
                           0,                    // lower bound
                           sizeof(float),        // extent
                           &block_type);
    
    MPI_Type_commit(&block_type);
    MPI_Type_free(&tmp_type);
    
    return block_type;
}

MPI_Comm create_cartesian_topology(MPI_Comm comm, int grid_size){
	int dims[2] = {grid_size, grid_size};
	int periods[2] = {0, 0};
	MPI_Comm comm_cart;
	MPI_Cart_create(comm, 2, dims, periods, 0, &comm_cart);
	return comm_cart;
}


float* scatter_matrix(float* matrix, int rank, int size, int m, int k, MPI_Comm comm){
	RowCol local_a_rc;
	int grid_size = (int)sqrt(size);
	if (rank == 0){
		local_a_rc.rows = ceil(m / grid_size);
		local_a_rc.cols = ceil(k / grid_size);
	}
	MPI_Datatype rowcol_type = create_rowcol_type();
	MPI_Bcast(&local_a_rc, 1, rowcol_type, 0, MPI_COMM_WORLD);
	MPI_Type_free(&rowcol_type);

	// Now send over the parts of A to each processor
	float* local_a = (float*)malloc(local_a_rc.rows * local_a_rc.cols * sizeof(float));

	scatter_row_major_matrix(matrix, local_a, m, k, grid_size, rank, size, comm);
	return local_a;
}

float* init_c_matrix_for_stationary_c(int m, int k, int n, int n_processors, int rank){
	int grid_size = (int)sqrt(n_processors);
	int block_m = ceil(m / grid_size);
	int block_n = ceil(n / grid_size);
	float* local_c = (float*)calloc(block_m * block_n, sizeof(float));
	return local_c;
}

void set_send_offset_for_block_scat_gath(int* sendcounts, int* displs, int m,
												int k, int grid_size, MPI_Comm comm) {
	int local_rows = m / grid_size;
	int local_cols = k / grid_size;
	for(int i = 0; i < grid_size * grid_size; i++) {
		sendcounts[i] = 1;
		int coords[2];
		MPI_Cart_coords(comm, i, 2, coords);
		int p_row = coords[0];
		int p_col = coords[1];
		displs[i] = p_row * local_rows * k + p_col * local_cols;
	}
}

void gather_row_major_matrix(float* local_matrix, float* global_matrix, 
					int m, int n, int grid_size, int rank, int size, MPI_Comm comm) {
	
	int *recvcounts = (int*)malloc(size * sizeof(int));
	int *displs = (int*)malloc(size * sizeof(int));

	int blocksize = m / grid_size * n / grid_size;
	set_send_offset_for_block_scat_gath(recvcounts, displs, m, n, grid_size, comm);

	// Create MPI datatype for the block
	MPI_Datatype blocktype = create_block_type(m, n, grid_size);

	// Gather the blocks
	MPI_Gatherv(local_matrix, blocksize, MPI_FLOAT,
				global_matrix, recvcounts, displs, blocktype,
				0, comm);

	// Cleanup
	MPI_Type_free(&blocktype);
	free(recvcounts);
	free(displs);
}