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
		//printf("rc.rows: %d\n", rc.rows);
		//printf("rc.cols: %d\n", rc.cols);
		assert(rc.rows == 4);
		assert(rc.cols == 4);
	}
	MPI_Type_free(&rowcol_type);
	if (rank == 0){
		printf("test_send_custom_data_type passed\n");
	}
}

void test_sending_a_to_processors_for_stationary_c_summa(int rank, int size){
	int m = 4;
	int k = 4;
	//int n = 8;
	int n_processors = 4;
	int grid_size = (int)sqrt(n_processors);

	// Create the grid of processors with MPI
	CartCommunicator cart_com = create_cartesian_topology(MPI_COMM_WORLD, grid_size);
	MPI_Comm comm = cart_com.cart_comm;
	// Generate the A and B matrices
	float *A = NULL;
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
	if (rank == 0){
		free(A);
	}
	free(local_a);
	MPI_Comm_free(&comm);
	MPI_Comm_free(&cart_com.parent_comm);
	if (rank == 0){
		printf("test_sending_a_to_processors_for_stationary_c_summa passed\n");
	}
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

void test_2d_grid(int rank, int size){
	MPI_Barrier(MPI_COMM_WORLD);
	if (size != 4){
		if (rank == 0){
			printf("Skipping test_2d_grid. Works with 4 processors\n");
		}
		return;
	}
	if (rank == 0){
		printf("Starting test_2d_grid\n");
	}
	int grid_size = 2;
	CartCommunicator cart_com = create_cartesian_topology(MPI_COMM_WORLD, grid_size);
	MPI_Comm comm = cart_com.cart_comm;
	int coords[2];
	MPI_Cart_coords(comm, rank, 2, coords);
	if (rank == 0){
		assert(coords[0] == 0);
		assert(coords[1] == 0);
	}
	if (rank == 1){
		assert(coords[0] == 0);
		assert(coords[1] == 1);
	}
	if (rank == 2){
		assert(coords[0] == 1);
		assert(coords[1] == 0);
	}
	if (rank == 3){
		assert(coords[0] == 1);
		assert(coords[1] == 1);
	}

	MPI_Comm_free(&comm);
	MPI_Comm_free(&cart_com.parent_comm);
	if (rank == 0){
		printf("test_2d_grid passed\n");
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

// When the processors are split in a 2D grid, check to see that
// we can gather a matrix across rows into a temp matrix on the 
// root column. 
void test_reduce_across_rows(int rank, int size){
	MPI_Barrier(MPI_COMM_WORLD);
	if (size != 4){
		if (rank == 0){
			printf("Skipping test_gather_across_rows. Works with 4 processors\n");
		}
		return;
	}
	if (rank == 0){
		printf("Starting test_gather_across_rows\n");
	}
	//printf("Starting test gather\n");

	int m = 4;
	int n = 8;
	int grid_size = (int)sqrt(size);

	// Create matrices where each element is the same as their rank
	float* A = generate_rank_matrix(m, n, rank);
	float* column_0_gathered_A = NULL;
	float* expected_gathered_A = NULL;

	// Create the grid of processors with MPI
	CartCommunicator cart_com = create_cartesian_topology(MPI_COMM_WORLD, grid_size);
	MPI_Comm comm = cart_com.cart_comm;

	// create the processor grid
	int coords[2];
	MPI_Cart_coords(comm, rank, 2, coords);
	MPI_Comm row_comm;
	MPI_Comm_split(comm, coords[0], coords[1], &row_comm);

	// Create the expected gathered A matrix
	if (coords[0] == 0 & coords[1] == 0){
		expected_gathered_A = generate_rank_matrix(m, n, 1);
	} else if (coords[0] == 1 && coords[1] == 0){
		expected_gathered_A = generate_rank_matrix(m, n, 5);
	}

	// Now gather the A matrix across the rows
	if (coords[1] == 0){
		column_0_gathered_A = (float*)calloc(m * n, sizeof(float));
	}
	
	// Gather across the row!
	MPI_Reduce(A, column_0_gathered_A, m * n, MPI_FLOAT, MPI_SUM, 0, row_comm);

	if (coords[1] == 0){
		printf("column_0_gathered_A rank %d\n", rank);
		print_matrix(column_0_gathered_A, m, n);
		printf("expected_gathered_A\n");
		print_matrix(expected_gathered_A, m, n);
	}

	// Now we need to verify the result
	if (coords[1] == 0){
		assert(do_matrices_match(column_0_gathered_A, expected_gathered_A, m, n, 0.001));
	}

	// Free data
	if (coords[1] == 0){
		free(column_0_gathered_A);
		free(expected_gathered_A);
	}
	free(A);
	MPI_Comm_free(&row_comm);
	MPI_Comm_free(&comm);
	MPI_Comm_free(&cart_com.parent_comm);
	
	if (rank == 0){
		printf("test_gather_across_rows passed\n");
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

// Rank 0 will have a matrix of dimensions grid_size. 
// Column 0 will gather a value into a specific column of the matrix
void test_gather_column(int rank, int size){
	int grid_size = (int)sqrt(size);
	float* A = NULL;
	if (rank == 0){
		A = (float*)calloc(grid_size, sizeof(float));
	}
	// Create the grid of processors with MPI
	CartCommunicator cart_com = create_cartesian_topology(MPI_COMM_WORLD, grid_size);
	MPI_Comm comm = cart_com.cart_comm;
	// create the processor grid
	int coords[2];
	MPI_Cart_coords(comm, rank, 2, coords);
	MPI_Comm row_comm, col_comm;
	MPI_Comm_split(comm, coords[0], coords[1], &row_comm);
	MPI_Comm_split(comm, coords[1], coords[0], &col_comm);

	// Create the column0 communicator
	MPI_Comm col_0_comm;
	int color = (coords[1] == 0) ? 0 : MPI_UNDEFINED;
	MPI_Comm_split(comm, color, coords[0], &col_0_comm);

	float val_to_gather = rank;
	// Now gather the values in column 0 into A on the root process
	if (coords[1] == 0){
		MPI_Gather(&val_to_gather, 1, MPI_FLOAT, A, 1, MPI_FLOAT, 0, col_0_comm);
	}

	// Now we need to verify the result
	float* expected_gathered_A = (float*)calloc(grid_size, sizeof(float));
	for (int i = 0; i < grid_size; i++){
		expected_gathered_A[i] = i * grid_size;
	}

	if (coords[1] == 0 && coords[0] == 0){
		printf("A matrix after gathering column 0\n");
		print_matrix(A, grid_size, 1);
		printf("expected_gathered_A\n");
		print_matrix(expected_gathered_A, grid_size, 1);
		printf("B matrix after gathering column 0\n");
		assert(do_matrices_match(A, expected_gathered_A, grid_size, 1, 0.001));
	}

	if (rank == 0){
		printf("test_gather_column passed\n");
	}
	// Clean up
	if (rank == 0){
		free(A);
	}
	free(expected_gathered_A);
	//MPI_Comm_free(&row_comm);
	//MPI_Comm_free(&col_comm);
	if (coords[1] == 0){
		MPI_Comm_free(&col_0_comm);
	}
	MPI_Comm_free(&comm);
	MPI_Comm_free(&cart_com.parent_comm);

}

// Tests gathering a matrix from a 2D grid of processors
// and placing it into the correct location of the matrix in rank 0
// ex:
// given column_to_place = 1 and the matrix is in row 1 of a 2x2 grid, 
// the matrix should be placed in the bottom right corner of the matrix
// on p0
void test_block_gather_column_into_matrix(int rank, int size){
	int m = 4;
	int n = 8;

	int grid_size = (int)sqrt(size);
	int local_m = ceil(m / grid_size);
	int local_n = ceil(n / grid_size);
	float* A = NULL;
	float* local_a = generate_rank_matrix(local_m, local_n, rank + 1);

	if (rank == 0){
		A = (float*)calloc(m * n, sizeof(float));
	}
	// Start by gathering the matrix from column 0 to the root processor
	float expected_matrix[32] = {
		1, 1, 1, 1, 0, 0, 0, 0,
		1, 1, 1, 1, 0, 0, 0, 0,
		3, 3, 3, 3, 0, 0, 0, 0,
		3, 3, 3, 3, 0, 0, 0, 0
	};
	// Create the grid of processors with MPI
	CartCommunicator cart_com = create_cartesian_topology(MPI_COMM_WORLD, grid_size);
	MPI_Comm comm = cart_com.cart_comm;
	// create the processor grid
	int coords[2];
	MPI_Cart_coords(comm, rank, 2, coords);
	MPI_Comm row_comm, col_comm;
	MPI_Comm_split(comm, coords[0], coords[1], &row_comm);
	MPI_Comm_split(comm, coords[1], coords[0], &col_comm);

	// Now gather the local A matrices into the correct position on the root processor
	int column_to_place = 0;
	MPI_Comm col_0_comm;
	int color = (coords[1] == 0) ? 0 : MPI_UNDEFINED;
	MPI_Comm_split(comm, color, coords[0], &col_0_comm); 
	if (coords[1] == 0){
		gather_col_blocks_into_root_matrix(local_a, A, m, n, grid_size, rank, size,
					column_to_place, comm, col_0_comm);
	}
	// Now we need to verify the result
	if (rank == 0){
		printf("A matrix after gathering column %d\n", column_to_place);
		print_matrix(A, m, n);
	}

	// Now try the full thing
	for (int i = 0; i < grid_size; i++){
		if(coords[1] == 0){
			gather_col_blocks_into_root_matrix(local_a, A, m, n, grid_size, rank, size,
					i, comm, col_0_comm);
		}
	}

	float full_expected[32] = {
		1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1,
		3, 3, 3, 3, 3, 3, 3, 3,
		3, 3, 3, 3, 3, 3, 3, 3
	};

	if (rank == 0){
		printf("A matrix after gathering all columns\n");
		print_matrix(A, m, n);
		printf("expected matrix\n");
		print_matrix(full_expected, m, n);
		assert(do_matrices_match(A, full_expected, m, n, 0.001));
	}

	// Clean up
	if (rank == 0){
		free(A);
	}
	free(local_a);
	MPI_Comm_free(&row_comm);
	MPI_Comm_free(&col_comm);
	MPI_Comm_free(&comm);
	MPI_Comm_free(&cart_com.parent_comm);
	if (rank == 0){
		printf("test_gather_column_into_matrix passed\n");
	}
}

// Given a 2D grid of processors, test to see if we can broadcast a matrix
// from row r, col c to all processors in column c
void test_bcast_p_along_column(int rank, int size){
	int m = 4;
	int n = 8;
	float* A = NULL;
	if (rank == 0){
		A = generate_int_matrix(m, n, 0);
	}
	int grid_size = (int)sqrt(size);
	int local_m = ceil(m / grid_size);
	int local_n = ceil(n / grid_size);

	// scatter the A matrix to the processors
	CartCommunicator cart_com = create_cartesian_topology(MPI_COMM_WORLD, grid_size);
	MPI_Comm comm = cart_com.cart_comm;
	// create the processor grid
	int coords[2];
	MPI_Cart_coords(comm, rank, 2, coords);
	
	float* local_a = scatter_matrix(A, rank, size, m, n, comm);
	float* tmp_a = (float*)malloc(local_m * local_n * sizeof(float));
	float* send_vals = (float*)malloc(local_m * local_n * sizeof(float));
	memcpy(tmp_a, local_a, local_m * local_n * sizeof(float));

	// let's test broadcasting A in 00 to all processors in column 0, 
	broadcast_matrix_to_column(local_a, send_vals, tmp_a, local_m * local_n, 0, 0, grid_size, rank, comm);
	
	MPI_Barrier(MPI_COMM_WORLD);
	broadcast_matrix_to_column(local_a, send_vals, tmp_a, local_m * local_n, 2, 1, grid_size, rank, comm);
	float expected[8] = {
		0, 1, 2, 3,
		8, 9, 10, 11
	};
	// Now we need to verify the result
	// if (coords[1] == 0){
	// 	printf("tmp_a after broadcast in column 0 rank %d\n", rank);
	// 	print_matrix(tmp_a, local_m, local_n);
	// }
	MPI_Barrier(MPI_COMM_WORLD);
	
	float col_1_expected[9] = {
		16, 17, 18, 19,
		24, 25, 26, 27
	};

	if (coords[1] == 1){
		//printf("tmp_a after broadcast in column 1 rank %d\n", rank);
		//print_matrix(tmp_a, local_m, local_n);
		assert(do_matrices_match(tmp_a, col_1_expected, local_m, local_n, 0.001));
	}
	else if (coords[1] == 0){
		assert(do_matrices_match(tmp_a, expected, local_m, local_n, 0.001));
	}

}

void test_stationary_a_summa(int rank, int size){
	if (rank == 0){
		printf("Testing stationary_a_summa\n");
	}
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

	if (rank == 0){
		printf("Pre scatter\n");
		printf("A matrix \n");
		print_matrix(A, m, k);
		printf("B matrix \n");
		print_matrix(B, k, n);
		printf("\n\n\n");
	}

	CartCommunicator cart_com = create_cartesian_topology(MPI_COMM_WORLD, grid_size);
	MPI_Comm comm = cart_com.cart_comm;

	// Distribute A and B chunks to the processors
	float* local_a = scatter_matrix(A, rank, size, m, k, comm);
	float* local_b = scatter_matrix(B, rank, size, k, n, comm);

	// create temp B and C matrices
	// These are the same size as the local_a and local_b matrices
	int local_a_rows = ceil(m / grid_size);
	int local_a_cols = ceil(k / grid_size);
	int local_b_rows = ceil(k / grid_size);
	int local_b_cols = ceil(n / grid_size);
	int local_c_rows = ceil(m / grid_size);
	int local_c_cols = ceil(n / grid_size);

	float* tmp_b = (float*)calloc(local_b_rows * local_b_cols, sizeof(float));
	float* tmp_c = (float*)calloc(local_c_rows * local_c_cols, sizeof(float));
	float* send_b = (float*)calloc(local_b_rows * local_b_cols, sizeof(float));

	MPI_Barrier(MPI_COMM_WORLD);
	// printf("local_b after scatter rank %d\n", rank);
	// print_matrix(local_b, local_b_rows, local_b_cols);

	// Now we need to broadcast B along the rows. 
	// Let's test just broadcasting B along the rows
	
	int coords[2];
	// Now broadcast the B from column 0 across the rows
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

	// // Broadcast B once
	// if (coords[1] == 0){
	// 	memcpy(tmp_b, local_b, local_b_rows * local_b_cols * sizeof(float));
	// }
	// MPI_Bcast(tmp_b, local_b_rows * local_b_cols, MPI_FLOAT, 0, row_comm);

	// // Now confirm that the broadcasted B is correct
	// float p_expected_b_matrix[8];
	// if (coords[0] == 0){
	// 	p_expected_b_matrix[0] = 0;
	// 	p_expected_b_matrix[1] = 1;
	// 	p_expected_b_matrix[2] = 2;
	// 	p_expected_b_matrix[3] = 3;
	// 	p_expected_b_matrix[4] = 8;
	// 	p_expected_b_matrix[5] = 9;
	// 	p_expected_b_matrix[6] = 10;
	// 	p_expected_b_matrix[7] = 11;
	// } else if (coords[0] == 1){
	// 	p_expected_b_matrix[0] = 4;
	// 	p_expected_b_matrix[1] = 5;
	// 	p_expected_b_matrix[2] = 6;
	// 	p_expected_b_matrix[3] = 7;
	// 	p_expected_b_matrix[4] = 12;
	// 	p_expected_b_matrix[5] = 13;
	// 	p_expected_b_matrix[6] = 14;
	// 	p_expected_b_matrix[7] = 15;
	// }
	// if (rank == 0){
	// 	printf("p_expected_b_matrix\n");
	// 	print_matrix(p_expected_b_matrix, 2, 4);
	// }

	// if (rank == 0 || rank == 1){
	// 	printf("tmp_b rank %d \n", rank);
	// 	print_matrix(tmp_b, local_b_rows, local_b_cols);
	// 	printf("expected_b\n");
	// }

	// if (rank == 0 || rank == 1){
	// 	assert(do_matrices_match(tmp_b, p_expected_b_matrix, local_b_rows, local_b_cols, 0.001));
	// }
	// if (rank == 0){
	// 	printf("tmp_b broadcasted correctly\n");
	// }


	//  let's do the full calculation
	int local_c_size = local_c_rows * local_c_cols;
	for (int c_col = 0; c_col < grid_size; c_col++){
	//for (int c_col = 0; c_col < 1; c_col++){
	// Broadcast the B from column i across the rows
		// if (coords[1] == c_col){
		// 	memcpy(tmp_b, local_b, local_b_rows * local_b_cols * sizeof(float));
		// }
		//MPI_Bcast(tmp_b, local_b_rows * local_b_cols, MPI_FLOAT, i, row_comm);
		
		for (int i = 0; i < grid_size; i++){
			// Now we need to broadcast the tmp_b matrix to the processors in the column
			// of the processor row
			// Root node is the processor in row j of column i
			int root = i * grid_size + c_col;
			// TODO -> check to see if the column that we are sending to is correct

			// The column needs to change with i!!!
			broadcast_matrix_to_column(local_b, send_b, tmp_b, local_b_rows * local_b_cols,
				root, i, grid_size, rank, comm);
		}

		// Print the tmp_b matrix to see if it is correct
		printf("tmp_b after broadcast in column %d rank %d\n", c_col, rank);
		print_matrix(tmp_b, local_b_rows, local_b_cols);
		printf("\n\n");

		// Clear the local c matrix and column 0 gathered c matrix
		memset(tmp_c, 0, local_c_size * sizeof(float));
		if (coords[1] == 0){
			memset(column_0_gathered_C, 0, local_c_size * sizeof(float));
		}

		// All row's have the b from column i now. Perform the matrix multiplication
		// into their local_c matrices
		matmul(local_a, tmp_b, tmp_c, local_a_rows, local_b_cols, local_a_cols);

		// Reduce onto column 0
		MPI_Reduce(tmp_c, column_0_gathered_C, local_c_size, MPI_FLOAT, MPI_SUM, 0, row_comm);

		// Now we need to place the gathered local_c matrix into the global C matrix
		if(coords[1] == 0){
			gather_col_blocks_into_root_matrix(column_0_gathered_C, C, m, n, grid_size, rank, size,
				c_col, comm, col_0_comm);
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}

	// Now we need to verify the result
	if (rank == 0){

		printf("A matrix \n");
		print_matrix(A, m, k);
		printf("B matrix \n");
		print_matrix(B, k, n);
		float* expected_c = (float*)calloc(m * n, sizeof(float));
		matmul(A, B, expected_c, m, n, k);
		printf("Expected = \n");
		print_matrix(expected_c, m, n);
		print_matrix(C, m, n);
		assert(do_matrices_match(C, expected_c, m, n, 0.001));
		free(expected_c);
	}


	// Free data
	if (rank == 0){
		free(A);
		free(B);
		free(C);
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
		printf("stationary_a_summa passed\n");
	}
}


int run_tests(int argc, char *argv[]) {
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	// test_2d_grid(rank, size);
	// test_reduce_across_rows(rank, size);
	// test_gather_column(rank, size);
	// test_block_gather_column_into_matrix(rank, size);
	// test_place_submatrix_into_full_matrix(rank, size);
	// test_send_custom_data_type(rank, size);
	// test_sending_a_to_processors_for_stationary_c_summa(rank, size);
	// test_stationary_c_summa(rank, size);
	// test_create_cartesian_topology(rank, size);
	// test_bcast_p_along_column(rank, size);
	
	test_stationary_a_summa(rank, size);

	MPI_Finalize();
	if (rank == 0){
		printf("All tests passed\n");
	}
	return 0;
}