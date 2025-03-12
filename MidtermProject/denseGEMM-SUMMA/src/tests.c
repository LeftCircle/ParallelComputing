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
void test_get_c_dimensions_for_stationary_c(){
	int m = 4;
	int k = 4;
	int n = 8;
	int n_processors = 4;

	Matrix *local_c = init_c_matrix_for_stationary_c(m, k, n, n_processors, 0);
	assert(local_c->rows == 2);
	assert(local_c->cols == 4);
	assert(sizeof(local_c->matrix) == 2*4*sizeof(float));
}


int run_tests() {
	//test_create_matrix();
	test_get_c_dimensions_for_stationary_c();
	printf("All tests passed\n");
	return 0;
}