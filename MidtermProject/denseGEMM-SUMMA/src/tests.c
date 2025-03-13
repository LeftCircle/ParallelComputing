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
	MPI_Comm comm = create_cartesian_topology(MPI_COMM_WORLD, grid_size);
	
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
}

void test_stationary_c_summa(int rank, int size){
	int m = 4;
	int k = 4;
	int n = 8;
	int grid_size = (int)sqrt(size);
	int n_processors = size;
	// Create the grid of processors with MPI
	MPI_Comm comm = create_cartesian_topology(MPI_COMM_WORLD, grid_size);

	// Generate the A and B matrices
	float *A;
	float *B;
	if (rank == 0){
		A = generate_int_matrix(m, k, 0);
		B = generate_int_matrix(k, n, 0);
	}

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


	// Now we need to broadcast A along the rows
	// Now we need to broadcast B along the columns
	float* tmp_a = (float*)malloc(local_rows * local_k * sizeof(float));
	float* tmp_b = (float*)malloc(local_k * local_cols * sizeof(float));
	for (int i = 0; i < grid_size; i++){
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

		// Multiply the matrices
		matmul(tmp_a, tmp_b, local_c, local_rows, local_cols, local_k);
	}

	// Now we need to gather the local C matrices to the global C matrix
	float* C = (float*)malloc(m * n * sizeof(float));
	gather_row_major_matrix(local_c, C, m, n, grid_size, rank, size, comm);

	// Now we need to verify the result
	if (rank == 0){
		verify_result(C, A, B, m, n, k);
	}

	free(A);
	free(B);
	free(local_a);
	free(local_b);
	free(local_c);
	free(C);
	free(tmp_a);
	free(tmp_b);
	MPI_Comm_free(&comm);
	MPI_Comm_free(&row_comm);
	MPI_Comm_free(&col_comm);


}


int run_tests(int argc, char *argv[]) {
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	//test_create_matrix();
	//test_get_c_dimensions_for_stationary_c();
	test_send_custom_data_type(rank, size);
	test_sending_a_to_processors_for_stationary_c_summa(rank, size);
	
	MPI_Finalize();
	printf("All tests passed\n");
	return 0;
}