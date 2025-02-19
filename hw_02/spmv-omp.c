// -----------------------------------------
// Richard Cato
// 2/11/2025
// NCSU CSC 548 Parallel Systems
// -----------------------------------------
// spmv parallelized with MPI. 
// Parallized the sequential code from
// https://github.ncsu.edu/jli256/CSC548/tree/main/Assignments/SpMV


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

int N_THREADS = 8;

int rank, size;
coo_bench_data og_coo_bench_data;

// split num_nonzeros between threads -> would not have parallel for
//
// Or move the int tid into the omp parallel for and remove the first #pragma omp parallel
void coo_spmv_omp(coo_matrix * coo, float * x, float * y,
					    float**y_thread, int num_threads){
	int num_nonzeros = coo->num_nonzeros;
	#pragma omp parallel for
	for (int i = 0; i < num_nonzeros; i++){   
		int tid = omp_get_thread_num();
		y_thread[tid][coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
	}
	// Reduce the results to the main y array
	// This reduction step is slow
	#pragma omp parallel for
	for (int i = 0; i < coo->num_rows; i++){
		for (int j = 0; j < num_threads; j++){
			y[i] += y_thread[j][i];
		}
	}
}

void coo_spmv_omp_reduction(coo_matrix * coo, float * x, float * y){
	#pragma omp parallel for reduction(+:y[:coo->num_rows]) num_threads(N_THREADS)
	for (int i = 0; i < coo->num_nonzeros; i++){   
		y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
	}
}

// // Could try using #pragma omp critical on the y addition to see if it is faster
// void coo_smpv_omp_single_y(coo_matrix * coo, float * x, float * y, float mod){
// 	int num_nonzeros = coo->num_nonzeros;
// 	#pragma omp parallel for shared(y, x, coo) reduction(+:y)
// 	for (int i = 0; i < num_nonzeros; i++){   
// 		y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]] * mod;
// 	}
// }


double benchmark_coo_smpv_omp(coo_matrix * coo, float* x, float* y, const char * mm_filename)
{
	int num_nonzeros = coo->num_nonzeros;

	// the y value is accessed by each thread, so it's better to 
	// give each thread their own and sum them in the end just like 
	// with MPI
	//float **y_thread = (float **)malloc(num_threads * sizeof(float *));

	float **y_thread = (float **)malloc(N_THREADS * sizeof(float *));
	for (int i = 0; i < N_THREADS; i++){
		y_thread[i] = (float*)calloc(coo->num_rows, sizeof(float));
		if (y_thread[i] == NULL){
			printf("Error allocating memory for y_thread\n");
			exit(1);
		}
	}

	timer time_one_iteration;
    timer_start(&time_one_iteration);
	
	// Now the parallel part
	coo_spmv_omp_reduction(coo, x, y);
	//coo_smpv_omp_single_y(coo, x, y);
	double estimated_time = seconds_elapsed(&time_one_iteration); 
	
	#ifdef SEQUENTIAL_CHECK
	// Test against sequential
		coo_matrix seq_coo;
		read_coo_matrix(&seq_coo, mm_filename);
		float * seq_x = (float*)malloc(seq_coo.num_cols * sizeof(float));
		float * seq_y = (float*)malloc(seq_coo.num_rows * sizeof(float));
		init_matrix_and_xy_vals(&seq_coo, seq_x, seq_y);
		coo_spmv(&seq_coo, seq_x, seq_y);
		test_spmv_accuracy(y, seq_y, seq_coo.num_rows, 0.01);
		free(seq_x);
		free(seq_y);
		delete_coo_matrix(&seq_coo);
	#endif

	// determine # of seconds dynamically
	int num_iterations;
	num_iterations = MAX_ITER;

	if (estimated_time == 0)
		num_iterations = MAX_ITER;
	else {
		num_iterations = min(MAX_ITER, max(MIN_ITER, (int) (TIME_LIMIT / estimated_time)) ); 
	}

	// time several SpMV iterations
    timer t;
    timer_start(&t);
    for(int j = 0; j < num_iterations; j++){
		coo_spmv_omp_reduction(coo, x, y);
	}
    double msec_per_iteration = milliseconds_elapsed(&t) / (double) num_iterations;
    double sec_per_iteration = msec_per_iteration / 1000.0;
	double GFLOPs = (sec_per_iteration == 0) ? 0 : (2.0 * (double) og_coo_bench_data.n_nonzeros / sec_per_iteration) / 1e9;
	double GBYTEs = (sec_per_iteration == 0) ? 0 : ((double) og_coo_bench_data.bytes_per_coospmv / sec_per_iteration) / 1e9;
	printf("\tbenchmarking COO-SpMV: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s)\n", msec_per_iteration, GFLOPs, GBYTEs); 

	// cleanup
	for(int i = 0; i < N_THREADS; i++){
		free(y_thread[i]);
	}
	free(y_thread);

    return msec_per_iteration;
}

int main(int argc, char** argv)
{
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (rank == 0){
		omp_set_num_threads(N_THREADS);
		
		#ifdef DEBUG
		#pragma omp parallel
		{
			int t_id = omp_get_thread_num();
			printf("Hello from thread %d of %d\n", t_id, omp_get_num_threads());
		}
		#endif

		if (get_arg(argc, argv, "help") != NULL){
			usage(argc, argv);
			return 0;
		}

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
		og_coo_bench_data.n_nonzeros = coo.num_nonzeros;
		og_coo_bench_data.bytes_per_coospmv = bytes_per_coo_spmv(&coo);

		x = (float *)malloc(coo.num_cols * sizeof(float));
		y = (float *)malloc(coo.num_rows * sizeof(float));
		init_matrix_and_xy_vals(&coo, x, y);
		
		double coo_gflops;
		coo_gflops = benchmark_coo_smpv_omp(&coo, x, y, mm_filename);
		printf("COO GFLOPS for OpenMP: %f\n", coo_gflops);

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

		
		// Now free the stuff
		free(x);
		free(y);
		delete_coo_matrix(&coo);
	}

	MPI_Finalize();
    return 0;
}





