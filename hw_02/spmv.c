#include "spmv.h"

// MIN_ITER, MAX_ITER, TIME_LIMIT, 
double benchmark_coo_spmv(coo_matrix * coo, float* x, float* y)
{
    int num_nonzeros = coo->num_nonzeros;

    // warmup    
    timer time_one_iteration;
    timer_start(&time_one_iteration);
    coo_spmv(coo, x, y);

    double estimated_time = seconds_elapsed(&time_one_iteration); 
	// printf("estimated time for once %f\n", (float) estimated_time);

    // determine # of seconds dynamically
    int num_iterations;
    num_iterations = MAX_ITER;

    if (estimated_time == 0)
        num_iterations = MAX_ITER;
    else {
        num_iterations = min(MAX_ITER, max(MIN_ITER, (int) (TIME_LIMIT / estimated_time)) ); 
    }
    printf("\tPerforming %d iterations\n", num_iterations);

    // time several SpMV iterations
    timer t;
    timer_start(&t);
    for(int j = 0; j < num_iterations; j++)
       coo_spmv(coo, x, y);
    double msec_per_iteration = milliseconds_elapsed(&t) / (double) num_iterations;
    double sec_per_iteration = msec_per_iteration / 1000.0;
    double GFLOPs = (sec_per_iteration == 0) ? 0 : (2.0 * (double) coo->num_nonzeros / sec_per_iteration) / 1e9;
    double GBYTEs = (sec_per_iteration == 0) ? 0 : ((double) bytes_per_coo_spmv(coo) / sec_per_iteration) / 1e9;
    printf("\tbenchmarking COO-SpMV: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s)\n", msec_per_iteration, GFLOPs, GBYTEs); 

    return msec_per_iteration;
}

void coo_spmv(coo_matrix* coo, float* x, float* y){
	int num_nonzeros = coo->num_nonzeros;
	for (int i = 0; i < num_nonzeros; i++){   
		y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
	}
}

void usage(int argc, char** argv)
{
    printf("Usage: %s [my_matrix.mtx]\n", argv[0]);
    printf("Note: my_matrix.mtx must be real-valued sparse matrix in the MatrixMarket file format.\n"); 
}

void init_matrix_and_xy_vals(coo_matrix * coo, float *x, float *y){
	srand(13);
	for (int i = 0; i < coo->num_nonzeros; i++){
		coo->vals[i] = 1.0 - 2.0 * (rand() / (RAND_MAX + 1.0)); 
	}
	for(int i = 0; i < coo->num_cols; i++) {
        x[i] = rand() / (RAND_MAX + 1.0); 
    }
    for(int i = 0; i < coo->num_rows; i++){
        y[i] = 0;
	}
}

int get_n_iterations(int estimated_time){
	int num_iterations;
	if (estimated_time == 0)
        num_iterations = MAX_ITER;
    else {
        num_iterations = min(MAX_ITER, max(MIN_ITER, (int) (TIME_LIMIT / estimated_time)) ); 
    }
	return num_iterations;
}