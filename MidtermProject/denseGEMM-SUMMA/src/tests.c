#include "tests.h"


// void test_create_matrix(){
// 	// Create an m x n matrix. 
// 	// It should just be a 1D array of size m*n
// 	float *matrix = generate_matrix(3, 4);
// 	// assert that all the values of the matrix are zero
// 	for (int i = 0; i < 3*4; i++){
// 		assert(matrix[i] == 0);
// 	}
// 	// assert that the matrix is of the correct size
// 	assert(sizeof(matrix) == 3*4*sizeof(float));
// 	free(matrix);
// 	printf("create_matrix passed\n");
// }

// Checks to see that the local C matrix dimensions are
// correct for stationary C SUMMA
// void test_get_c_dimensions_for_stationary_c(){
// 	int m = 4;
// 	int k = 4;
// 	int n = 8;
// 	int n_processors = 4;
// 	int grid_size = (int)sqrt(n_processors);


// 	Matrix *local_c = init_c_matrix_for_stationary_c(m, k, n, grid_size, 0);
// 	printf("local_c->rows: %d\n", local_c->rows);
// 	printf("local_c->cols: %d\n", local_c->cols);

// 	assert(local_c->rows == 2);
// 	assert(local_c->cols == 4);
// }

void test_send_custom_data_type(int rank, int size){
	RowCol rc;
	MPI_Datatype rowcol_type = create_rowcol_type();
	if (rank == 0){
		rc.rows = 4;
		rc.cols = 4;
	}
	MPI_Bcast(&rc, 1, rowcol_type, 0, MPI_COMM_WORLD);
	if (rank == 1){
		printf("rc.rows: %d\n", rc.rows);
		printf("rc.cols: %d\n", rc.cols);
		assert(rc.rows == 4);
		assert(rc.cols == 4);
	}
	MPI_Type_free(&rowcol_type);
}

void test_sending_a_to_processors_for_stationary_c_summa(int rank, int size){
	int m = 4;
	int k = 4;
	int n = 8;
	int n_processors = 4;
	int grid_size = (int)sqrt(n_processors);

	// Create the grid of processors with MPI
	CartCommunicator cart_com = create_cartesian_topology(MPI_COMM_WORLD, grid_size);
	MPI_Comm comm = cart_com.cart_comm;
	// Generate the A and B matrices
	float *A;
	//float *B;
	if (rank == 0){
		A = generate_int_matrix(m, k, 0);
		//B = generate_int_matrix(k, n, 0);
	}

	
	float p_expected_a_matrix[4];
	if (rank == 0){
		p_expected_a_matrix[0] = 0;
		p_expected_a_matrix[1] = 1;
		p_expected_a_matrix[2] = 4;
		p_expected_a_matrix[3] = 5;
	} else if (rank == 1){
		p_expected_a_matrix[0] = 2;
		p_expected_a_matrix[1] = 3;
		p_expected_a_matrix[2] = 6;
		p_expected_a_matrix[3] = 7;
	} else if (rank == 2){
		p_expected_a_matrix[0] = 8;
		p_expected_a_matrix[1] = 9;
		p_expected_a_matrix[2] = 12;
		p_expected_a_matrix[3] = 13;
	} else if (rank == 3){
		p_expected_a_matrix[0] = 10;
		p_expected_a_matrix[1] = 11;
		p_expected_a_matrix[2] = 14;
		p_expected_a_matrix[3] = 15;
	}

	// scatter_row_major_matrix(A, local_a, m, k, grid_size, rank, size, comm);
	float* local_a = scatter_matrix(A, rank, size, m, k, comm);

	int local_a_rows = ceil(m / grid_size);
	int local_a_cols = ceil(k / grid_size);
	// Now check that the local_a matrix is correct
	for (int i = 0; i < local_a_rows * local_a_cols; i++){
		assert(local_a[i] == p_expected_a_matrix[i]);
	}

	free(A);
	free(local_a);
	MPI_Comm_free(&comm);
	MPI_Comm_free(&cart_com.parent_comm);
}

void test_stationary_c_summa(int rank, int size){
	int m = 4;
	int k = 4;
	int n = 8;
	
	float* C = stationary_c_summa(m, k, n, rank, size);

	// Now we need to verify the result
	if (rank == 0){
		float* A = generate_int_matrix(m, k, 0);
		float* B = generate_int_matrix(k, n, 0);
		float * expected_c = (float*)calloc(m * n, sizeof(float));
		matmul(A, B, expected_c, m, n, k);
		printf("Expected = \n");
		print_matrix(expected_c, m, n);
		print_matrix(C, m, n);
		for (int i = 0; i < m * n; i++){
			assert(abs(C[i]) - abs(expected_c[i]) < 0.001);
		}
		free(expected_c);
		free(A);
		free(B);
		free(C);
		printf("stationary_c_summa passed\n");
	}
}

void test_create_cartesian_topology(int rank, int size){
	int grid_size = 2;
	CartCommunicator cart_com = create_cartesian_topology(MPI_COMM_WORLD, grid_size);
	MPI_Comm comm = cart_com.cart_comm;
	int coords[2];
	MPI_Cart_coords(comm, rank, 2, coords);
	int status;
	MPI_Topo_test(comm, &status);
	assert(status == MPI_CART);
	if (rank == 0){
		printf("Cartesian topo test passed\n");
	}
	MPI_Comm_free(&comm);
	MPI_Comm_free(&cart_com.parent_comm);
}

void test_place_submatrix_into_full_matrix(int rank, int size){
	if (rank == 0){
		int m = 4;
		int n = 4;
		int local_m = 2;
		int local_n = 2;
		// sub_position is row, col
		int sub_position[2] = {1, 1};
		float* full_matrix = (float*)calloc(m * n, sizeof(float));
		float* sub_matrix = (float*)malloc(local_m * local_n * sizeof(float));
		for (int i = 0; i < local_m * local_n; i++){
			sub_matrix[i] = i;
		}
		place_submatrix_into_full_matrix(full_matrix, sub_matrix, m, n, local_m, local_n, &sub_position);
		float expected_full_matrix[16] = {0, 0, 0, 0,
										  0, 0, 0, 0,
										  0, 0, 0, 1,
										  0, 0, 2, 3};
		assert(do_matrices_match(full_matrix, expected_full_matrix, m, n, 0.001));
		free(full_matrix);
		free(sub_matrix);
		printf("place_submatrix_into_full_matrix passed\n");
	}
}

void test_stationary_a_summa(int rank, int size){
	int m = 4;
	int k = 4;
	int n = 8;
	int grid_size = (int)sqrt(size);

	// Create the test matrices
	float* A = NULL;
	float* B = NULL;
	float* C = NULL;
	if (rank == 0){
		A = generate_int_matrix(m, k, 0);
		B = generate_int_matrix(k, n, 0);
		C = (float*)calloc(m * n, sizeof(float));
	}

	CartCommunicator cart_com = create_cartesian_topology(MPI_COMM_WORLD, grid_size);
	MPI_Comm comm = cart_com.cart_comm;

	// Distribute A and B chunks to the processors
	float* local_a = scatter_matrix(A, rank, size, m, k, comm);
	float* local_b = scatter_matrix(B, rank, size, k, n, comm);

	// create temp B and C matrices
	// These are the same size as the local_a and local_b matrices
	
	int local_b_rows = ceil(k / grid_size);
	int local_b_cols = ceil(n / grid_size);
	int local_c_rows = ceil(m / grid_size);
	int local_c_cols = ceil(n / grid_size);

	float* tmp_b = (float*)calloc(local_b_rows * local_b_cols, sizeof(float));
	float* tmp_c = (float*)calloc(local_c_rows * local_c_cols, sizeof(float));

	// Now we need to broadcast B along the rows. 
	// Let's test just broadcasting B along the rows
	
	
	int coords[2];
	// Now broadcast the B from column 0 across the rows
	MPI_Cart_coords(comm, rank, 2, coords);
	MPI_Comm row_comm, col_comm;
	MPI_Comm_split(comm, coords[0], coords[1], &row_comm);
	MPI_Comm_split(comm, coords[1], coords[0], &col_comm);

	// Broadcast B once
	if (coords[1] == 0){
		memcpy(tmp_b, local_b, local_b_rows * local_b_cols * sizeof(float));
	}
	MPI_Bcast(tmp_b, local_b_rows * local_b_cols, MPI_FLOAT, 0, row_comm);

	// Now confirm that the broadcasted B is correct
	float p_expected_b_matrix[8];
	if (coords[0] == 0){
		p_expected_b_matrix[0] = 0;
		p_expected_b_matrix[1] = 1;
		p_expected_b_matrix[2] = 2;
		p_expected_b_matrix[3] = 3;
		p_expected_b_matrix[4] = 8;
		p_expected_b_matrix[5] = 9;
		p_expected_b_matrix[6] = 10;
		p_expected_b_matrix[7] = 11;
	} else if (coords[0] == 1){
		p_expected_b_matrix[0] = 4;
		p_expected_b_matrix[1] = 5;
		p_expected_b_matrix[2] = 6;
		p_expected_b_matrix[3] = 7;
		p_expected_b_matrix[4] = 12;
		p_expected_b_matrix[5] = 13;
		p_expected_b_matrix[6] = 14;
		p_expected_b_matrix[7] = 15;
	}
	if (rank == 0){
		printf("p_expected_b_matrix\n");
		print_matrix(p_expected_b_matrix, 2, 4);
	}

	if (rank == 0 || rank == 1){
		printf("tmp_b rank %d \n", rank);
		print_matrix(tmp_b, local_b_rows, local_b_cols);
		printf("expected_b\n");
	}

	if (rank == 0 || rank == 1){
	assert(do_matrices_match(tmp_b, p_expected_b_matrix, local_b_rows, local_b_cols, 0.001));
	}

	// Now that we know the row broadcast works, let's do the full calculation

	float* rank_0_gather_local_c = NULL;
	if (rank == 0){
		rank_0_gather_local_c = (float*)calloc(local_c_cols * local_c_rows, sizeof(float));
	}
	for (int i = 0; i < grid_size; i++){
		if (coords[1] == i){
			memcpy(tmp_b, local_b, local_b_rows * local_b_cols * sizeof(float));
		}
		MPI_Bcast(tmp_b, local_b_rows * local_b_cols, MPI_FLOAT, i, row_comm);
		
		// Reset the gather matrix on rank 0
		if (rank == 0){
			memset(rank_0_gather_local_c, 0, local_c_rows * local_c_cols * sizeof(float));
		}
		// Clear the local c matrix
		memset(tmp_c, 0, local_c_rows * local_c_cols * sizeof(float));

		// All row's have the b from column i now. Perform the matrix multiplication
		// into their local_c matrices
		matmul(local_a, tmp_b, tmp_c, local_c_rows, local_c_cols, local_b_rows);

		// Now we need to gather the local C matrices to the global C matrix
		gather_matrix_across_rows(tmp_c, rank_0_gather_local_c, m, n, grid_size, rank, size, row_comm);

		// Now we need to place the gathered local_c matrix into the global C matrix
		if (rank == 0){

			// If we are to gather this way, it would require gathering each row, placing that into
			// rank_0_gather_local_c, then placing that into the global C matrix. 
			// There might be a better way to Gatherv the local_c matrices into the global C matrix
			assert(false);
		}
		

	}


	// Free data
	if (rank == 0){
		free(A);
		free(B);
		free(C);
	}
	free(local_a);
	free(local_b);
	free(tmp_b);
	free(tmp_c);

	MPI_Comm_free(&comm);
	MPI_Comm_free(&row_comm);
	MPI_Comm_free(&col_comm);
	MPI_Comm_free(&cart_com.parent_comm);

	if (rank == 0){
		printf("stationary_a_summa passed\n");
	}
}


int run_tests(int argc, char *argv[]) {
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	test_place_submatrix_into_full_matrix(rank, size);
	test_send_custom_data_type(rank, size);
	test_sending_a_to_processors_for_stationary_c_summa(rank, size);
	test_stationary_c_summa(rank, size);
	test_create_cartesian_topology(rank, size);
	//test_stationary_a_summa(rank, size);

	MPI_Finalize();
	if (rank == 0){
		printf("All tests passed\n");
	}
	return 0;
}