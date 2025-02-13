// A set of useful parallel functions
// -----------------------------------------
// Richard Cato
// 2/12/2025
// NCSU CSC 548 Parallel Systems
// -----------------------------------------

#pragma once
#include <stdio.h>


static inline void split_workload(int n, int p, int *workload_array, int *workload_displ){
	int workload = n/p;
	int excess = n%p;
	for (int i = 0; i < p; i++){
		workload_array[i] = workload;
		if (i < excess){
			workload_array[i]++;
		}
		workload_displ[i] = workload_array[i] * sizeof(int);
	}
}

static inline void print_veci(int * vec, int size){
	for (int i = 0; i < size; i++){
		printf(" %d ", vec[i]);
	}
	printf("\n");
}

static inline void print_vecf(float * vec, int size){
	for (int i = 0; i < size; i++){
		printf(" %d ", vec[i]);
	}
	printf("\n");
}