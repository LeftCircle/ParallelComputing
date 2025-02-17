#ifndef SPMV_MPI_H
#define SPMV_MPI_H

#include <stdio.h>
#include <mpi.h>

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

// MPI specific functions
void _rank_zero_startup(coo_matrix * coo, float **x, float **y, const char * mm_filename);
void _rank_zero_data_scatter(coo_matrix *coo);
void _other_rank_data_scatter(coo_matrix * coo);
void _split_vals_between_nodes(coo_matrix * coo);
void _broadcast_data_for_x_and_y(coo_matrix * coo, float **x, float **y);
void _coo_spmv_mpi(coo_matrix *coo, float *x, float *y, float * y_parallel);
void _init_x_and_y_for_nonzero_nodes(coo_matrix *coo, float **x, float **y);


#endif