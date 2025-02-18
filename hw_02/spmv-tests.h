#ifndef SPMV_TESTS_H
#define SPMV_TESTS_H

#include <assert.h>
#include <stdio.h>



static inline void test_spmv_accuracy(float * parallel_y, float * sequential_y, int size, float threshold){
	// check to see that the values of x allign
	for (int i = 0; i < size; i++){
		if (parallel_y[i] - sequential_y[i] > threshold){
			printf("ERROR Index %d does not match: actual %f vs expected %f\n", i, parallel_y[i], sequential_y[i]);
		}
		assert(parallel_y[i] - sequential_y[i] <= threshold);
	}
	
	printf("--------------\nParallel and sequential match!!\n--------------\n");
}

#endif