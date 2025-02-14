// A set of tests to make sure everything is working as expected

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <omp.h>

#include "parallelFuncs.h"


// test split up workload
// Given n elements and p processors, each processor should get n/p elements
// and the eccess elements should be distributed among the starting processors
void test_split_workload(){
	// Testing using floats
	int n = 13;
	int p = 3;
	
	int *workload_array = (int *)malloc(p * sizeof(int));
	int *workload_offset = (int*)malloc(p * sizeof(int));
	split_workload(n, p, workload_array, workload_offset);

	assert(workload_array[0] == 5);
	assert(workload_array[1] == 4);
	assert(workload_array[2] == 4);
	assert(workload_offset[0] == 0);
	assert(workload_offset[1] == 5);
	assert(workload_offset[2] == 9);
	free(workload_array);
	free(workload_offset);
	printf("test_split_workload passed\n");
}

void test_openmp_for_loop(){
	int max_threads = omp_get_max_threads();
	int index_per_thread = 2;
	int loop_size = max_threads * index_per_thread;
	int *results = (int*)calloc(loop_size, sizeof(int));

	#pragma omp parallel for
	for (int i = 0; i < loop_size; i++){
		int thread_id = omp_get_thread_num();
		results[i] = thread_id;
	}
	// Check that the results are correct
	for (int i = 0; i < loop_size; i++){
		assert(results[i] == i / index_per_thread);
	}

	free(results);
	printf("Test omp for passed\n");
}

void test_uneven_omp_for_loop(){
	int max_threads = omp_get_max_threads();
	int index_per_thread = 2;
	int loop_size = max_threads * index_per_thread + max_threads / 2;
	int *results = (int*)calloc(loop_size, sizeof(int));

	#pragma omp parallel for
	for (int i = 0; i < loop_size; i++){
		int thread_id = omp_get_thread_num();
		results[i] = thread_id;
	}
	print_veci(results, loop_size);
	// Check that the results are correct

	free(results);
	printf("Test uneven omp for passed\n");
}

void test_openmp_for_loop_reduction(){
	int max_threads = omp_get_max_threads();
	int index_per_thread = 2;
	int loop_size = max_threads * index_per_thread;
	int *results = (int*)calloc(loop_size, sizeof(int));

	#pragma omp parallel for reduction(+:results[:loop_size])
	for (int i = 0; i < loop_size; i++){
		results[i] = i;
	}
	// Check that the results are correct
	for (int i = 0; i < loop_size; i++){
		assert(results[i] == i);
	}
	//print_veci(results, loop_size);
	free(results);
	printf("Test omp for reduction passed\n");
}

int main(int argc, char *argv[]){
	test_split_workload();
	test_openmp_for_loop();
	test_openmp_for_loop_reduction();
	printf("All tests passed\n");
	return 0;
}
