// A set of tests to make sure everything is working as expected

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#include "parallelFuncs.h"


// test split up workload
// Given n elements and p processors, each processor should get n/p elements
// and the eccess elements should be distributed among the starting processors
void test_split_workload(){
	int n = 13;
	int p = 3;
	
	int *workload_array = (int *)malloc(p * sizeof(int));
	split_workload(n, p, workload_array);

	assert(workload_array[0] == 5);
	assert(workload_array[1] == 4);
	assert(workload_array[2] == 4);
	free(workload_array);
	printf("test_split_workload passed\n");
}

int main(int argc, char *argv[]){
	test_split_workload();
	printf("All tests passed\n");
	return 0;
}
