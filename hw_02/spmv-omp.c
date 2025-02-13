// -----------------------------------------
// Richard Cato
// 2/11/2025
// NCSU CSC 548 Parallel Systems
// -----------------------------------------
// spmv parallelized with MPI. 
// Parallized the sequential code from
// https://github.ncsu.edu/jli256/CSC548/tree/main/Assignments/SpMV


#include <stdio.h>
#include <omp.h>

#include "cmdline.h"
#include "input.h"
#include "config.h"
#include "timer.h"
#include "formats.h"
#include "parallelFuncs.h"

#define max(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a > _b ? _a : _b; })

#define min(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a < _b ? _a : _b; })

void usage(int argc, char** argv)
{
    printf("Usage: %s [my_matrix.mtx]\n", argv[0]);
    printf("Note: my_matrix.mtx must be real-valued sparse matrix in the MatrixMarket file format.\n"); 
}

// MIN_ITER, MAX_ITER, TIME_LIMIT, 
double benchmark_coo_spmv(coo_matrix * coo, float* x, float* y)
{
    int num_nonzeros = coo->num_nonzeros;

    // warmup    
    timer time_one_iteration;
    timer_start(&time_one_iteration);
    for (int i = 0; i < num_nonzeros; i++){   
        y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
    }

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
        for (int i = 0; i < num_nonzeros; i++){   
            y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]] * 0;
        }
    double msec_per_iteration = milliseconds_elapsed(&t) / (double) num_iterations;
    double sec_per_iteration = msec_per_iteration / 1000.0;
    double GFLOPs = (sec_per_iteration == 0) ? 0 : (2.0 * (double) coo->num_nonzeros / sec_per_iteration) / 1e9;
    double GBYTEs = (sec_per_iteration == 0) ? 0 : ((double) bytes_per_coo_spmv(coo) / sec_per_iteration) / 1e9;
    printf("\tbenchmarking COO-SpMV: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s)\n", msec_per_iteration, GFLOPs, GBYTEs); 

    return msec_per_iteration;
}

double benchmark_coo_smpv_omp(coo_matrix * coo, float* x, float* y)
{
	int num_nonzeros = coo->num_nonzeros;
	int num_rows = coo->num_rows;

	// Get number of threads
	int num_threads = omp_get_max_threads();

	// the y value is accessed by each thread, so it's better to 
	// give each thread their own and sum them in the end just like 
	// with MPI
	float **y_thread = (float **)malloc(num_threads * sizeof(float *));
	
	// Now the parallel part
	#pragma omp parallal
	{
		int tid = omp_get_thread_num();
		// calloc to init to 0
		y_thread[tid] = (float *)calloc(num_rows, sizeof(float));
		#pragma omp for
		// benchmark the spmv
		double coo_gflops;
		coo_gflops = benchmark_coo_spmv(&coo, x, y_thread[tid]);
	}

	#pragma omp parallel for
	for (int i = 0; i < num_rows; i++){
		for (int j = 0; j < num_threads; j++){
			y[i] += y_thread[j][i];
		}
	}
	// Free the local y arrays
	for (int i = 0; i < num_threads; i++){
		free(y_thread[i]);
	}
	free(y_thread);
	return 0;
}

void init_matrix_and_xy_vals(coo_matrix * coo, float * x, float * y){
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

int main(int argc, char** argv)
{
	if (get_arg(argc, argv, "help") != NULL){
		usage(argc, argv);
        return 0;
    }

	// Only rank 0 will read the matrix
    char * mm_filename = NULL;
    if (argc == 1) {
        printf("Give a MatrixMarket file.\n");
        return -1;
    } else { 
        mm_filename = argv[1];
	}

	// The size of the arrays that node 0 will send
	coo_matrix coo;
	float * x;
	float * y;

	//coo_matrix coo;
	read_coo_matrix(&coo, mm_filename);


	x = (float *)malloc(coo.num_cols * sizeof(float));
	y = (float *)malloc(coo.num_rows * sizeof(float));
	init_matrix_and_xy_vals(&coo, x, y);
	
	double coo_gflops;
	coo_gflops = benchmark_coo_smpv_omp(&coo, x, y);

    /* Test correctnesss */
	#ifdef TESTING
		printf("Writing x and y vectors ...");
		FILE *fp = fopen("test_x", "w");
		for (int i=0; i<coo.num_cols; i++)
		{
		fprintf(fp, "%f\n", x[i]);
		}
		fclose(fp);
		fp = fopen("test_y", "w");
		for (int i=0; i<coo.num_rows; i++)
		{
		fprintf(fp, "%f\n", y[i]);
		}
		fclose(fp);
		printf("... done!\n");
	#endif

	#ifdef SEQUENTIAL_CHECK
	// Test against sequential
		coo_matrix seq_coo;
		read_coo_matrix(&seq_coo, mm_filename);
		float * seq_x = (float*)malloc(seq_coo.num_cols * sizeof(float));
		float * seq_y = (float*)malloc(seq_coo.num_rows * sizeof(float));
		init_matrix_and_xy_vals(&seq_coo, seq_x, seq_y);
		double coo_gflops_s = benchmark_coo_spmv(&seq_coo, seq_x, seq_y);
		test_spmv_accuracy(y, seq_y, seq_coo.num_rows, 0.01);
	#endif
	// Now free the stuff
	free(x);
	free(y);
	delete_coo_matrix(&coo);
    return 0;
}





