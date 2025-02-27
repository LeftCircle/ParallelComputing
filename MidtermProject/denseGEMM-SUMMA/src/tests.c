#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#include "utils.h"

void test_create_matrix(){
	// Create an m x n matrix. 
	// It should just be a 1D array of size m*n
	float *matrix = generate_matrix(3, 4);
	// assert that all the values of the matrix are zero
	for (int i = 0; i < 3*4; i++){
		assert(matrix[i] == 0);
	}
	// assert that the matrix is of the correct size
	assert(sizeof(matrix) == 3*4*sizeof(float));
	free(matrix);
	printf("create_matrix passed\n");
}



int main() {
	test_create_matrix();
	printf("All tests passed\n");
	return 0;
}