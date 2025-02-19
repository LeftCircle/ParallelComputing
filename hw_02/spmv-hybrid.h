#ifndef SPMV_HYBRID_H
#define SPMV_HYBRID_H

#include <stdio.h>
#include <mpi.h>
#include <omp.h>

#include "cmdline.h"
#include "input.h"
#include "config.h"
#include "timer.h"
#include "formats.h"
#include "parallelFuncs.h"
#include "spmv-tests.h"
#include "spmv.h"

// Global variables for MPI
int rank;
int size;

typedef struct spmv_args{
	coo_matrix * coo;
	float ** x;
	float ** y;
	float * y_parallel;
} spmv_args;

// MPI specific functions
void _rank_zero_startup(coo_matrix * coo, float **x, float **y, float **y_parallel, const char * mm_filename);
void _rank_zero_data_scatter(coo_matrix *coo);
void _other_rank_data_scatter(coo_matrix * coo);
void _split_vals_between_nodes(coo_matrix * coo);
void _broadcast_data_for_x_and_y(coo_matrix * coo, float **x, float **y);
void _coo_spmv_mpi_omp(coo_matrix *coo, float **x, float **y, float *y_parallel);
void _init_x_and_y_for_nonzero_nodes(coo_matrix *coo, float **x, float **y);
void _time_without_data_transfer(coo_matrix *coo, float **x, float **y, float *y_parallel);

// OMP
void coo_spmv_omp(coo_matrix * coo, float * x, float * y);

// wrapper functions
static inline void spmv_wrapper(void* args){
	coo_matrix * coo = ((spmv_args*)args)->coo;
	float ** x = ((spmv_args*)args)->x;
	float ** y = ((spmv_args*)args)->y;
	float * y_parallel = ((spmv_args*)args)->y_parallel;
	_coo_spmv_mpi_omp(coo, x, y, y_parallel);
}

/**
 * Generic function timer for MPI programs
 * @param func_to_time Function pointer to the function to be timed
 * @param args Void pointer to arguments structure
 * @return Execution time in seconds
 */
double time_function_ms(int n_iterations, void (*func_to_time)(void*), void* args);

#endif