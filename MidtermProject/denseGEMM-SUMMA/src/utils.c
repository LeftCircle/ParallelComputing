#include "utils.h"


void matmul(float* A, float* B, float* C, int m, int n, int k) {
    // Initialize output matrix to zero
    //memset(C, 0, m * n * sizeof(float));
	// C[i,j] = sum(A[i,p] * B[p,j])
	for (int i = 0; i < m; i++) {
		for (int p = 0; p < k; p++) {
			for (int j = 0; j < n; j++) {
				C[i * n + j] += A[i * k + p] * B[p * n + j];
			}
		}
	}
}

void verify_result(float* C_global, float* C_ref, float* A, float* B, int m, int n, int k) {
    int errors = 0;
    float tolerance = 1e-5;
    // Perform reference matrix multiplication
    // float* C_ref = (float*)calloc(m * n, sizeof(float));
    // if (!C_ref) {
    //     printf("Error: Failed to allocate memory for C_ref\n");
    //     return;
    // }

    // // Compute reference result
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < n; j++) {
    //         float sum = 0.0f;
    //         for (int p = 0; p < k; p++) {
    //             sum += A[i*k + p] * B[p*n + j];
    //         }
    //         C_ref[i*n + j] = sum;
    //     }
    // }

    // Compute detailed error statistics
    float max_error = 0.0f;
    float avg_error = 0.0f;
    int max_error_index = -1;
    int max_error_i = -1, max_error_j = -1;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i*n + j;
			// Compute the reference result
			// for (int p = 0; p < k; p++) {
            //     C_ref[idx] += A[i*k + p] * B[p*n + j];
            // }
            float curr_error = fabs(C_global[idx] - C_ref[idx]);
            avg_error += curr_error;
            
            if (curr_error > max_error) {
                max_error = curr_error;
                max_error_index = idx;
                max_error_i = i;
                max_error_j = j;
            }
            
            if (curr_error > tolerance) {
                errors++;   
				// if (errors <= 5) {
                //     printf("Error at position [%d,%d]: C_global=%.6f, C_ref=%.6f, diff=%.6f\n",
                //            i, j, C_global[idx], C_ref[idx], curr_error);
                // }
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

    //free(C_ref);
}

// Utility function to load matrix from file or generate
float* generate_matrix_A(int rows, int cols, int rank) {
    if (rank == 0){

		float* matrix = malloc(rows * cols * sizeof(float));
		srand(42);
		
		for (int i = 0; i < rows * cols; i++) {
			matrix[i] = (float)rand() / RAND_MAX;
		}
		
		return matrix;
	}
	return NULL;
}


float* generate_matrix_B(int rows, int cols, int rank) {
	if (rank == 0){
		float* matrix = malloc(rows * cols * sizeof(float));
		srand(142);

		for (int i = 0; i < rows * cols; i++) {
			matrix[i] = (float)rand() / RAND_MAX;
		}

		return matrix;
	}
return NULL;
}

float* generate_int_matrix(int rows, int cols, int rank) {
	float* matrix = malloc(rows * cols * sizeof(float));
	
	for (int i = 0; i < rows * cols; i++) {
		matrix[i] = i;
	}
	
	return matrix;
}

float* generate_rank_matrix(int rows, int cols, int rank) {
	float* matrix = malloc(rows * cols * sizeof(float));
	
	for (int i = 0; i < rows * cols; i++) {
		matrix[i] = rank;
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
	int *sendcounts = (int*)malloc(size * sizeof(int));
	int *displs = (int*)malloc(size * sizeof(int));
	
	int local_rows = m / grid_size;
	int local_cols = k / grid_size;
	int blocksize = local_rows * local_cols;  // Size of each block

	int coords[2];

	set_send_offset_for_block_scat_gath(sendcounts, displs, m, k, grid_size, comm);

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

CartCommunicator create_cartesian_topology(MPI_Comm comm, int grid_size){
	int size;
	MPI_Comm_size(comm, &size);
	if (size != grid_size * grid_size){
		printf("Error: Number of processors must be a square number\n");
		MPI_Abort(comm, 1);
	}
	CartCommunicator cart_comm;
	MPI_Comm dup_comm;
	MPI_Comm_dup(comm, &dup_comm);
	int dims[2] = {grid_size, grid_size};
	int periods[2] = {0, 0};
	MPI_Comm comm_cart;
	int result = MPI_Cart_create(dup_comm, 2, dims, periods, 0, &comm_cart);
	if (result != MPI_SUCCESS){
		printf("Error: MPI_Cart_create failed with error code %d\n", result);
		MPI_Abort(dup_comm, result);
	}
	cart_comm.cart_comm = comm_cart;
	cart_comm.parent_comm = dup_comm;
	return cart_comm;
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

void set_send_offset_for_row_block_gatherv(int* sendcounts, int* displs, int p_col,
	 								int m, int n, int grid_size, MPI_Comm comm) {
	int local_rows = ceil(m / (float)grid_size);
    int local_cols = ceil(n / (float)grid_size);
	for(int i = 0; i < grid_size; i++) {
		sendcounts[i] = 1;
		int coords[2];
		MPI_Cart_coords(comm, i, 2, coords);
		int p_row = coords[0];
		// We are gathering into the same column, so p_col is constant
		displs[i] = p_row * local_rows * n + p_col * local_cols;
	}
}

// Requires the cart communicator. Cannot use the row communicator here
void set_send_offset_for_col_block_gatherv(int* sendcounts, int* displs, int p_col,
										int m, int n, int grid_size, MPI_Comm comm){
	int local_rows = ceil(m / (float)grid_size);
	int local_cols = ceil(n / (float)grid_size);
	for(int i = 0; i < grid_size; i++) {
		sendcounts[i] = 1;
		//int coords[2];
		//MPI_Cart_coords(comm, i, 2, coords);
		//int p_row = coords[0];
		int p_row = i;
		displs[i] = p_row * local_rows * n + p_col * local_cols;
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
	int result = MPI_Gatherv(local_matrix, blocksize, MPI_FLOAT,
				global_matrix, recvcounts, displs, blocktype,
				0, comm);
	
	if (result != MPI_SUCCESS) {
		printf("Error: MPI_Gatherv failed with error code %d\n", result);
	}

	// Cleanup
	MPI_Type_free(&blocktype);
	free(recvcounts);
	free(displs);
}

// Given a column of the global matrix to gather into, gathers the matrix
// from processors in column 0 into the column i of the global matrix
void gather_col_blocks_into_root_matrix(float* local_matrix, float* global_matrix,
	int m, int n, int grid_size, int rank, int size, int col, MPI_Comm cart_comm, MPI_Comm row_comm){
	
	int * recvcounts = (int*)malloc(grid_size * sizeof(int));
	int * displs = (int*)calloc(grid_size, sizeof(int));
	
	// Need to send the cart comm
	set_send_offset_for_col_block_gatherv(recvcounts, displs, col, m, n,
										grid_size, cart_comm);
	// if (rank == 0){
	// 	printf("Displacements = \n");
	// 	for (int i = 0; i < grid_size; i++){
	// 		printf("%d ", displs[i]);
	// 	}
	// 	printf("\n");
	// }
	// Create MPI datatype for the block
	MPI_Datatype blocktype = create_block_type(m, n, grid_size);

	// Gather the blocks if column 0
	int coords[2];
	MPI_Cart_coords(cart_comm, rank, 2, coords);
	if (coords[1] == 0){
		int result = MPI_Gatherv(local_matrix, m / grid_size * n / grid_size,
				MPI_FLOAT, global_matrix, recvcounts, displs, blocktype,
				0, row_comm);
		// if (result != MPI_SUCCESS) {
		// 	printf("Error: MPI_Gatherv failed with error code %d\n", result);
		// }
	}
	// Cleanup
	MPI_Type_free(&blocktype);
	free(recvcounts);
	free(displs);
}

void print_matrix(float* matrix, int rows, int cols) {
	printf("Printing matrix\n");
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			printf("%.2f ", matrix[i * cols + j]);
		}
		printf("\n");
	}
	printf("\n");
}

float* stationary_c_summa(int m, int k, int n, int rank, int size){
	int grid_size = (int)sqrt(size);
	int n_processors = size;
	// We only support cases where the matrices can be evenly divided
	if (m % grid_size != 0 || k % grid_size != 0){
		printf("Error: Matrix dimensions must be divisible by the grid size\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	MPITiming prof_time= create_mpi_timer();
	double init_start, init_end, comp_start, comp_end, comm_start, comm_end, total_start, total_end;
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0){
		total_start = MPI_Wtime();
		init_start = total_start;
	}
	
	// Generate the A and B matrices
	float *A = NULL;
	float *B = NULL;
	#ifdef UNIT_TESTING
	if (rank == 0){
		A = generate_int_matrix(m, k, 0);
		B = generate_int_matrix(k, n, 0);
	}
	#else
	if (rank == 0){
		A = generate_matrix_A(m, k, 0);
		B = generate_matrix_B(k, n, 0);
	}
	#endif
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0){
		init_end = MPI_Wtime();
		prof_time.init_time = init_end - init_start;
		comm_start = init_end;
	}

	// Create the grid of processors with MPI
	CartCommunicator cart_comm = create_cartesian_topology(MPI_COMM_WORLD, grid_size);
	MPI_Comm comm = cart_comm.cart_comm;

	// Create the local A and B matrices
	float* local_a = scatter_matrix(A, rank, size, m, k, comm);
	float* local_b = scatter_matrix(B, rank, size, k, n, comm);

	// Create the local C matrix
	float *local_c = init_c_matrix_for_stationary_c(m, k, n, n_processors, rank);

	// Create row and column communicators
	int coords[2];
	MPI_Cart_coords(comm, rank, 2, coords);
	MPI_Comm row_comm, col_comm;
	MPI_Comm_split(comm, coords[0], coords[1], &row_comm);
	MPI_Comm_split(comm, coords[1], coords[0], &col_comm);

	int local_rows = ceil(m / grid_size);
	int local_cols = ceil(n / grid_size);
	int local_k = ceil(k / grid_size);

	float* tmp_a = (float*)malloc(local_rows * local_k * sizeof(float));
	float* tmp_b = (float*)malloc(local_k * local_cols * sizeof(float));

	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0){
		comm_end = MPI_Wtime();
		prof_time.comm_time = comm_end - comm_start;
	}
	

	// Now we need to broadcast A along the rows
	// Now we need to broadcast B along the columns
	
	for (int i = 0; i < grid_size; i++){
		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0){
			comm_start = MPI_Wtime();
		}
		// Broadcast A
		if (coords[1] == i){
			memcpy(tmp_a, local_a, local_rows * local_k * sizeof(float));
		}
		MPI_Bcast(tmp_a, local_rows * local_k, MPI_FLOAT, i, row_comm);

		// Broadcast B
		if (coords[0] == i){
			memcpy(tmp_b, local_b, local_k * local_cols * sizeof(float));
		}
		MPI_Bcast(tmp_b, local_k * local_cols, MPI_FLOAT, i, col_comm);

		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0){
			comm_end = MPI_Wtime();
			prof_time.comm_time += comm_end - comm_start;
			comp_start = comm_end;
		}

		// Multiply the matrices
		matmul(tmp_a, tmp_b, local_c, local_rows, local_cols, local_k);

		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0){
			comp_end = MPI_Wtime();
			prof_time.comp_time += comp_end - comp_start;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	if(rank == 0){
		comm_start = MPI_Wtime();
	}

	// Now we need to gather the local C matrices to the global C matrix
	float* C = (float*)malloc(m * n * sizeof(float));
	gather_row_major_matrix(local_c, C, m, n, grid_size, rank, size, comm);

	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0){
		comm_end = MPI_Wtime();
		prof_time.comm_time += comm_end - comm_start;
		total_end = MPI_Wtime();
		prof_time.total_time = total_end - total_start;
		printf("Profiling times for stationary_c_summa\n");
		printf("m = %d, k = %d, n = %d with grid size %d\n", m, k, n, grid_size);
		printf("Initialization time: %f\n", prof_time.init_time);
		printf("Communication time: %f\n", prof_time.comm_time);
		printf("Computation time: %f\n", prof_time.comp_time);
		printf("Total time: %f\n", prof_time.total_time);
	}

	// Now we need to verify the result
	if (rank == 0){
		free(A);
		free(B);
	}

	free(local_a);
	free(local_b);
	free(local_c);
	free(tmp_a);
	free(tmp_b);
	MPI_Comm_free(&comm);
	MPI_Comm_free(&row_comm);
	MPI_Comm_free(&col_comm);
	MPI_Comm_free(&cart_comm.parent_comm);
	if (rank == 0){
		return C;
	}else{
		free(C);
		return NULL;
	}
}

bool do_matrices_match(float* A, float* B, int rows, int cols, float tolerance){
	for (int i = 0; i < rows * cols; i++){
		if (fabs(A[i] - B[i]) > tolerance){
			printf("Error: A[%d] = %.2f, B[%d] = %.2f\n", i, A[i], i, B[i]);
			return false;
		}
	}
	return true;
}

// Sub position is [row, col]
void place_submatrix_into_full_matrix(float* full_matrix, float* sub_matrix, int m,
									  int n, int local_m, int local_n, int* sub_position){
	int start_row = sub_position[0] * local_m;
	int start_col = sub_position[1] * local_n;
	int start_idx = start_row * n + start_col;
	
	//displs[i] = p_row * local_rows * k + p_col * local_cols;
	for (int j = 0; j < local_n; j++){
		for (int i = 0; i < local_m; i++){
			int full_idx = start_idx + i * n + j;
			int sub_idx = i * local_n + j;
			full_matrix[full_idx] = sub_matrix[sub_idx];
		}
	}
	//print_matrix(full_matrix, m, n);
}

bool is_in_array(int* array, int size, int value){
	for (int i = 0; i < size; i++){
		if (array[i] == value){
			return true;
		}
	}
	return false;
}

void broadcast_matrix_to_column(float* send_vals, float* send_buff, float* rcv_buff,
	 int count, int from_rank, int to_col, int grid_size, int rank, MPI_Comm comm){
	
	// if (rank == from_rank){
	// 	printf("Sending to column %d from rank %d\n", to_col, rank);
	// 	print_matrix(send_vals, 2, 4);
	// }
	// Get coordinates of current rank and sender
	int coords[2];
    int from_rank_coords[2];
	MPI_Cart_coords(comm, rank, 2, coords);
	MPI_Cart_coords(comm, from_rank, 2, from_rank_coords);

    
    // Create communicator for target column
    MPI_Comm col_comm;
    int color = (coords[1] == to_col) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(comm, color, coords[0], &col_comm);
    
    // Only processes in the target column participate
    if (col_comm != MPI_COMM_NULL) {
        // If sender is in column, use direct broadcast
        if (from_rank_coords[1] == to_col) {
            // Find sender's position in column communicator
            int bcast_root;
            int from_coords[2];
            MPI_Cart_coords(comm, from_rank, 2, from_coords);
            bcast_root = from_coords[0];  // Row coordinate is rank in column
			//printf("Bcast root is %d\n", bcast_root);
            //printf("To col is %d, coords[1] is %d\n", to_col, coords[1]);
            // Copy data to receive buffer if we're the sender
            if (rank == from_rank) {
                memcpy(rcv_buff, send_vals, count * sizeof(float));
            }
            
            // Broadcast within column
            MPI_Bcast(rcv_buff, count, MPI_FLOAT, bcast_root, col_comm);
        }
        // If sender not in column, rank 0 of column receives and broadcasts
        else {
            if (coords[0] == 0) {  // First process in column
                MPI_Recv(rcv_buff, count, MPI_FLOAT, from_rank, 0, comm, MPI_STATUS_IGNORE);
            }
            MPI_Bcast(rcv_buff, count, MPI_FLOAT, 0, col_comm);
        }
        
        MPI_Comm_free(&col_comm);
    }
    
    // If this is the sender and not in target column, send to first process in column
    if (rank == from_rank && coords[1] != to_col) {
        MPI_Send(send_vals, count, MPI_FLOAT, to_col, 0, comm);
    }
    
    MPI_Barrier(comm);  // Use cart comm instead of WORLD
}

float* stationary_a_summa(int m, int k, int n, int rank, int size){
	// Create the test matrices
	int grid_size = (int)sqrt(size);
	// We only support cases where the matrices can be evenly divided
	if (m % grid_size != 0 || k % grid_size != 0){
		printf("Error: Matrix dimensions must be divisible by the grid size\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	MPITiming prof_times = create_mpi_timer();
	double comm_start, comm_end, comp_start, comp_end, total_start, total_end, init_start, init_end;
	if (rank == 0){
		init_start = MPI_Wtime();
		total_start = init_start;
	}
	// Initialize matrices
	float* A = NULL;
	float* B = NULL;
	float* C = NULL;
	#ifdef UNIT_TESTING
	if (rank == 0){
		A = generate_int_matrix(m, k, 0);
		B = generate_int_matrix(k, n, 0);
		C = (float*)calloc(m * n, sizeof(float));
	}
	#else
	if (rank == 0){
		A = generate_matrix_A(m, k, 0);
		B = generate_matrix_B(k, n, 0);
		C = (float*)calloc(m * n, sizeof(float));
	}
	#endif

	// End initialization
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0){
		init_end = MPI_Wtime();
		prof_times.init_time = init_end - init_start;
		comm_start = init_end;
	}

	// Start matrix distribution
	CartCommunicator cart_com = create_cartesian_topology(MPI_COMM_WORLD, grid_size);
	MPI_Comm comm = cart_com.cart_comm;

	// Distribute A and B chunks to the processors
	float* local_a = scatter_matrix(A, rank, size, m, k, comm);
	float* local_b = scatter_matrix(B, rank, size, k, n, comm);
	// End matrix distribution

	// init sub matrix distributions
	int local_a_rows = ceil(m / grid_size);
	int local_a_cols = ceil(k / grid_size);
	int local_b_rows = ceil(k / grid_size);
	int local_b_cols = ceil(n / grid_size);
	int local_c_rows = ceil(m / grid_size);
	int local_c_cols = ceil(n / grid_size);
	
	// create temp B and C matrices
	float* tmp_b = (float*)calloc(local_b_rows * local_b_cols, sizeof(float));
	float* tmp_c = (float*)calloc(local_c_rows * local_c_cols, sizeof(float));
	float* send_b = (float*)calloc(local_b_rows * local_b_cols, sizeof(float));

	MPI_Barrier(MPI_COMM_WORLD);
	
	// Create row and column communicators
	int coords[2];
	MPI_Cart_coords(comm, rank, 2, coords);
	MPI_Comm row_comm, col_comm, col_0_comm;
	MPI_Comm_split(comm, coords[0], coords[1], &row_comm);
	MPI_Comm_split(comm, coords[1], coords[0], &col_comm);
	int color = (coords[1] == 0) ? 0 : MPI_UNDEFINED;
	MPI_Comm_split(comm, color, coords[0], &col_0_comm); 

	// Column 0 will reduce the local_c matrices into a temp matrix
	float* column_0_gathered_C = NULL;
	if (coords[1] == 0){
		column_0_gathered_C = (float*)calloc(local_c_rows * local_c_cols, sizeof(float));
	}
	// End distribution and creation of local matrices
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0){
		comm_end = MPI_Wtime();
		prof_times.comm_time = comm_end - comm_start;
	}

	//  let's do the full calculation
	int local_c_size = local_c_rows * local_c_cols;
	for (int c_col = 0; c_col < grid_size; c_col++){
		
		// Broadcast the B matrix to the column corrisponding to the b matrix row
		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0){
			comm_start = MPI_Wtime();
		}
		for (int i = 0; i < grid_size; i++){
			int root = i * grid_size + c_col;
			broadcast_matrix_to_column(local_b, send_b, tmp_b, local_b_rows * local_b_cols,
				root, i, grid_size, rank, comm);
		}

		// Clear the local c matrix and column 0 gathered c matrix
		memset(tmp_c, 0, local_c_size * sizeof(float));
		if (coords[1] == 0){
			memset(column_0_gathered_C, 0, local_c_size * sizeof(float));
		}
		// I'm including this in comm times since it wouldn't happen in a real implementation
		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0){
			comm_end = MPI_Wtime();
			prof_times.comm_time += comm_end - comm_start;
			comp_start = comm_end;
		}

		// Perform local calculation
		matmul(local_a, tmp_b, tmp_c, local_a_rows, local_b_cols, local_a_cols);

		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0){
			comp_end = MPI_Wtime();
			prof_times.comp_time += comp_end - comp_start;
			comm_start = comp_end;
		}

		// Reduce onto column 0
		MPI_Reduce(tmp_c, column_0_gathered_C, local_c_size, MPI_FLOAT, MPI_SUM, 0, row_comm);

		// Now we need to place the gathered local_c matrix into the global C matrix
		if(coords[1] == 0){
			gather_col_blocks_into_root_matrix(column_0_gathered_C, C, m, n, grid_size, rank, size,
				c_col, comm, col_0_comm);
		}

		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == 0){
			comm_end = MPI_Wtime();
			prof_times.comm_time += comm_end - comm_start;
		}
	}

	// Free data
	if (rank == 0){
		free(A);
		free(B);
	}
	if (coords[1] == 0){
		free(column_0_gathered_C);
	}
	// Free the local matrices
	free(local_a);
	free(local_b);
	free(tmp_b);
	free(tmp_c);

	if (coords[1] == 0){
		MPI_Comm_free(&col_0_comm);
	}
	MPI_Comm_free(&comm);
	MPI_Comm_free(&row_comm);
	MPI_Comm_free(&col_comm);
	MPI_Comm_free(&cart_com.parent_comm);
	if (rank == 0){
		prof_times.total_time = MPI_Wtime() - total_start;
		printf("Profiling times for stationary_A_summa\n");
		printf("m = %d, k = %d, n = %d with grid size %d\n", m, k, n, grid_size);
		printf("Initialization time: %f\n", prof_times.init_time);
		printf("Communication time: %f\n", prof_times.comm_time);
		printf("Computation time: %f\n", prof_times.comp_time);
		printf("Total time: %f\n", prof_times.total_time);
		return C;
	} else {
		return NULL;
	}
}


double time_function(void (*func)(int m, int k, int n, int rank, int size),
                    int m, int k, int n, int rank, int size) {
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize before timing
    double start_time = MPI_Wtime();
    
    func(m, k, n, rank, size);
    
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize after execution
    double end_time = MPI_Wtime();
    
    return end_time - start_time;
}

void print_timing_results(double local_time, int rank, const char* func_name) {
    double max_time, min_time, avg_time;
    
    // Gather timing statistics
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        avg_time /= size;
        printf("Timing results for %s:\n", func_name);
        printf("  Max time: %f seconds\n", max_time);
        printf("  Min time: %f seconds\n", min_time);
        printf("  Avg time: %f seconds\n", avg_time);
    }
}

void run_stationary_a_and_c_for(int m, int k, int n, int rank, int size, bool verify){
	MPI_Barrier(MPI_COMM_WORLD);
	float* C_stat_a = stationary_a_summa(m, k, n, rank, size);
	MPI_Barrier(MPI_COMM_WORLD);
	printf("\n");
	float* C_stat_c = stationary_c_summa(m, k, n, rank, size);
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0){
		if (verify){
			float* A = generate_matrix_A(m, k, rank);
			float* B = generate_matrix_B(k, n, rank);
			float* C_ref = (float*)calloc(m * n, sizeof(float));
			double start_time = MPI_Wtime();
			matmul(A, B, C_ref, m, n, k);
			double end_time = MPI_Wtime();
			printf("Time for reference matrix multiplication: %f\n", end_time - start_time);
			printf("Verifying stationary A:\n");
			verify_result(C_stat_a, C_ref, A, B, m, n, k);
			printf("\nVerifying stationary C:\n");
			verify_result(C_stat_c, C_ref, A, B, m, n, k);
			free(C_ref);
			free(A);
			free(B);
		}
		free(C_stat_a);
		free(C_stat_c);
	}
	MPI_Barrier(MPI_COMM_WORLD);
}


void print_mpi(int rank, const char* message){
	if (rank == 0){
		printf("%s\n", message);
	}
}

MPITiming create_mpi_timer(){
	MPITiming timer;
	timer.init_time = 0.0;
	timer.comm_time = 0.0;
	timer.comp_time = 0.0;
	timer.total_time = 0.0;
	return timer;
}