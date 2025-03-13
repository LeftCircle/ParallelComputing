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
	MPI_Comm comm;
	int dims[2] = {grid_size, grid_size};
	int periods[2] = {0, 0};
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm);
	
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

	
	// process 0 needs to send the correct parts of A to each processor
	RowCol local_a_rc;
	if (rank == 0){
		local_a_rc.rows = ceil(m / grid_size);
		local_a_rc.cols = ceil(k / grid_size);
	}
	MPI_Datatype rowcol_type = create_rowcol_type();
	MPI_Bcast(&local_a_rc, 1, rowcol_type, 0, MPI_COMM_WORLD);
	// Now send over the parts of A to each processor
	float* local_a = (float*)malloc(local_a_rc.rows * local_a_rc.cols * sizeof(float));

	
	// int* workload_array_size = (int*)malloc(size * sizeof(int));
	// int* workload_array_offset = (int*)malloc(size * sizeof(int));
	// for (int i = 0; i < size; i++){
	// 	workload_array_size[i] = local_a_rc.rows * local_a_rc.cols;
	// }
	// for (int i )
	scatter_row_major_matrix(A, local_a, m, k, grid_size, rank, size, comm);

	if (rank == 1){
		printf("local_a_rc.rows: %d\n", local_a_rc.rows);
		printf("local_a_rc.cols: %d\n", local_a_rc.cols);
		for (int i = 0; i < local_a_rc.rows * local_a_rc.cols; i++){
			printf("local_a[%d]: %f\n", i, local_a[i]);
		}
	}

	// Now check that the local_a matrix is correct
	for (int i = 0; i < local_a_rc.rows * local_a_rc.cols; i++){
		assert(local_a[i] == p_expected_a_matrix[i]);
	}

	MPI_Type_free(&rowcol_type);


	free(A);
	free(local_a);
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