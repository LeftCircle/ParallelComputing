#ifndef SPMV_H
#define SPMV_H

#include <stdio.h>

#include "cmdline.h"
#include "input.h"
#include "config.h"
#include "timer.h"
#include "formats.h"

#define max(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a > _b ? _a : _b; })

#define min(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a < _b ? _a : _b; })

typedef struct coo_bench_data{
	int n_nonzeros;
	double bytes_per_coospmv; 
} coo_bench_data;

// MIN_ITER, MAX_ITER, TIME_LIMIT, 
double benchmark_coo_spmv(coo_matrix * coo, float* x, float* y);
void coo_spmv(coo_matrix* coo, float* x, float* y);
void usage(int argc, char** argv);
void init_matrix_and_xy_vals(coo_matrix * coo, float *x, float *y);
int get_n_iterations(int estimated_time);

#endif
