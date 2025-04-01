
#include <stdio.h>
#include <cuda_runtime.h>
#include <mpi.h>

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

// Global variables for MPI
int rank;
int size;
coo_bench_data og_coo_bench_data;

void usage(int argc, char** argv)
{
    printf("Usage: %s [my_matrix.mtx]\n", argv[0]);
    printf("Note: my_matrix.mtx must be real-valued sparse matrix in the MatrixMarket file format.\n"); 
}

static inline void split_workload(int n, int p, int *workload_array, int *workload_displ){
	int workload = n/p;
	int excess = n%p;
	workload_array[0] = workload;
	if (0 < excess){
		workload_array[0]++;
	}
	workload_displ[0] = 0;
	for (int i = 1; i < p; i++){
		workload_array[i] = workload;
		if (i < excess){
			workload_array[i]++;
		}
		workload_displ[i] = workload_array[i - 1] + workload_displ[i - 1];
	}
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

void _rank_zero_startup(coo_matrix * coo, float **x, float **y,
	float **y_parallel, const char * mm_filename){
	//coo_matrix coo;
	read_coo_matrix(coo, mm_filename);
	*x = (float*)malloc(coo->num_cols * sizeof(float));
	*y = (float*)malloc(coo->num_rows * sizeof(float));
	*y_parallel = (float*)calloc(coo->num_rows, sizeof(float));
	init_matrix_and_xy_vals(coo, *x, *y);
}

void _rank_zero_data_scatter(coo_matrix *coo){
// Now we have to distribute the data. 
// Each node will get the entire y array, a full x array that will
// later be summed up, and a portion of the row/column/value from the
// coo array
	int *workload_array_size = (int *)malloc(size * sizeof(int));
	int *workload_displsi = (int*)malloc(size * sizeof(int));
	split_workload(coo->num_nonzeros, size, workload_array_size, workload_displsi);

	MPI_Scatter(workload_array_size, 1, MPI_INT, &coo->num_nonzeros, 1, MPI_INT,
	0, MPI_COMM_WORLD);

	MPI_Scatterv(coo->rows, workload_array_size, workload_displsi, MPI_INT,
				coo->rows, coo->num_nonzeros, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatterv(coo->cols, workload_array_size, workload_displsi, MPI_INT,
				coo->cols, coo->num_nonzeros, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatterv(coo->vals, workload_array_size, workload_displsi, MPI_FLOAT,
				coo->vals, coo->num_nonzeros, MPI_FLOAT, 0, MPI_COMM_WORLD);

	free(workload_array_size);
	free(workload_displsi);
}

void _other_rank_data_scatter(coo_matrix * coo){
	// Recieve workload size
	MPI_Scatter(NULL, 1, MPI_INT, &coo->num_nonzeros, 1, MPI_INT,
	0, MPI_COMM_WORLD);

	#ifdef DEBUG
	printf("Rank %d got workload size %d\n. Allocating space", rank, coo->num_nonzeros);
	#endif
	// Allocate space for the values
	coo->rows = (int*)malloc(coo->num_nonzeros * sizeof(int));
	coo->cols = (int*)malloc(coo->num_nonzeros * sizeof(int));
	coo->vals = (float*)malloc(coo->num_nonzeros * sizeof(float));

	// Now receive the buffers for this nodes portion of work
	MPI_Scatterv(NULL, NULL, NULL, MPI_INT, coo->rows, coo->num_nonzeros,
	MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatterv(NULL, NULL, NULL, MPI_INT, coo->cols, coo->num_nonzeros,
	MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatterv(NULL, NULL, NULL, MPI_FLOAT, coo->vals, coo->num_nonzeros,
	MPI_FLOAT, 0, MPI_COMM_WORLD);
}

void _split_vals_between_nodes(coo_matrix * coo){
	if (rank == 0){
		_rank_zero_data_scatter(coo);
	} else{
		_other_rank_data_scatter(coo);
	}
}

void _init_x_and_y_for_nonzero_nodes(coo_matrix *coo, float **x, float **y){
	*x = (float *)malloc(coo->num_cols * sizeof(float));
	*y = (float *)calloc(coo->num_rows, sizeof(float));
}

void _broadcast_data_for_x_and_y(coo_matrix *coo, float **x, float **y){
	MPI_Bcast(&coo->num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&coo->num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (rank != 0){
		_init_x_and_y_for_nonzero_nodes(coo, x, y);
	}
	// Now we can send the x array with the data in it
	MPI_Bcast(*x, coo->num_cols, MPI_FLOAT, 0, MPI_COMM_WORLD);
}


void coo_spmv(coo_matrix* coo, const float* x, float* y) {
	for (int i = 0; i < coo->num_nonzeros; i++){   
		y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
	}
}

__global__ void coo_spmv_kernel(int num_nonzeros, const int* rows, const int* cols, 
								const float* vals, const float* x, float* y) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < num_nonzeros) {
		atomicAdd(&y[rows[idx]], vals[idx] * x[cols[idx]]);
	}
}

void send_coo_spmv_data_to_gpu(coo_matrix* coo, float* x, int** d_rows, int** d_cols,
	float** d_vals, float** d_x, float** d_y) {
	// Allocate device memory
	cudaMalloc(d_rows, coo->num_nonzeros * sizeof(int));
	cudaMalloc(d_cols, coo->num_nonzeros * sizeof(int));
	cudaMalloc(d_vals, coo->num_nonzeros * sizeof(float));
	cudaMalloc(d_x, coo->num_cols * sizeof(float));
	cudaMalloc(d_y, coo->num_rows * sizeof(float));

	// Copy data to device
	cudaMemcpy(*d_rows, coo->rows, coo->num_nonzeros * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(*d_cols, coo->cols, coo->num_nonzeros * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(*d_vals, coo->vals, coo->num_nonzeros * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(*d_x, x, coo->num_cols * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemset(*d_y, 0, coo->num_rows * sizeof(float));  // Initialize output to zero
	cudaDeviceSynchronize();
}

double time_coo_smpv_mpi_kernal(coo_matrix* coo, float** y, float *y_parallel, int* d_rows, int* d_cols,
							float* d_vals, float* d_x, float* d_y) {
	// Launch kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (coo->num_nonzeros + threadsPerBlock - 1) / threadsPerBlock;
	timer t;
	int num_iterations = 500;
	timer_start(&t);

	for (int i = 0; i < num_iterations; i++){
		coo_spmv_kernel<<<blocksPerGrid, threadsPerBlock>>>(
			coo->num_nonzeros, d_rows, d_cols, d_vals, d_x, d_y
		);
		// wait for kernel to finish
		cudaDeviceSynchronize();
	}
	cudaMemcpy(*y, d_y, coo->num_rows * sizeof(float), cudaMemcpyDeviceToHost);
	MPI_Reduce(*y, y_parallel, coo->num_rows, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
	double msec_per_iteration = milliseconds_elapsed(&t) / (double) num_iterations;
	double sec_per_iteration = msec_per_iteration / 1000.0;
	double GFLOPs = (sec_per_iteration == 0) ? 0 : (2.0 * (double) coo->num_nonzeros / sec_per_iteration) / 1e9;
	double GBYTEs = (sec_per_iteration == 0) ? 0 : ((double) bytes_per_coo_spmv(coo) / sec_per_iteration) / 1e9;
	
	if (rank == 0){
		printf("\tbenchmarking COO-SpMV: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s)\n", msec_per_iteration, GFLOPs, GBYTEs);
	}
	return msec_per_iteration;
}

void coo_spmv_cuda(coo_matrix* coo, float* x, float* y) {
	// Device memory pointers
	int *d_rows, *d_cols;
	float *d_vals, *d_x, *d_y;
	send_coo_spmv_data_to_gpu(coo, x, &d_rows, &d_cols, &d_vals, &d_x, &d_y);

	// Launch kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (coo->num_nonzeros + threadsPerBlock - 1) / threadsPerBlock;

	coo_spmv_kernel<<<blocksPerGrid, threadsPerBlock>>>(
		coo->num_nonzeros, d_rows, d_cols, d_vals, d_x, d_y
	);

	// wait for kernel to finish
	cudaDeviceSynchronize();

	// Copy result back to host
	cudaMemcpy(y, d_y, coo->num_rows * sizeof(float), cudaMemcpyDeviceToHost);

	// Cleanup
	cudaFree(d_rows);
	cudaFree(d_cols);
	cudaFree(d_vals);
	cudaFree(d_x);
	cudaFree(d_y);
}

void benchmark_coo_spmv(coo_matrix* coo, float** x, float** y, float* y_parallel){
	// Device memory pointers
	_split_vals_between_nodes(coo);
	_broadcast_data_for_x_and_y(coo, x, y);
	int *d_rows, *d_cols;
	float *d_vals, *d_x, *d_y;
	send_coo_spmv_data_to_gpu(coo, *x, &d_rows, &d_cols, &d_vals, &d_x, &d_y);

	double time = time_coo_smpv_mpi_kernal(coo, y, y_parallel, d_rows, d_cols, d_vals, d_x, d_y);

	// Cleanup
	if (d_rows != NULL) cudaFree(d_rows);
	if (d_cols != NULL) cudaFree(d_cols);
	if (d_vals != NULL) cudaFree(d_vals);
	if (d_x != NULL) cudaFree(d_x);
	if (d_y != NULL) cudaFree(d_y);
	if (rank != 0){
		if (*x != NULL) free(*x);
		if (*y != NULL) free(*y);
	}
}

void coo_spmv_mpi_cuda(coo_matrix * coo, float **x, float **y, float * y_parallel){
	_split_vals_between_nodes(coo);
	_broadcast_data_for_x_and_y(coo, x, y);
	coo_spmv_cuda(coo, *x, *y);
	MPI_Reduce(*y, y_parallel, coo->num_rows, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
}

int main(int argc, char** argv)
{

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (get_arg(argc, argv, "help") != NULL){
        usage(argc, argv);
        return 0;
    }

    char * mm_filename = NULL;
    if (argc == 1) {
        printf("Give a MatrixMarket file.\n");
        return -1;
    } else 
        mm_filename = argv[1];

    coo_matrix coo;
	float * x;
	float * y;
	float * y_parallel;
	int og_nonzeros;
	if (rank == 0){
		_rank_zero_startup(&coo, &x, &y, &y_parallel, mm_filename);
		og_nonzeros = coo.num_nonzeros;
		og_coo_bench_data.n_nonzeros = coo.num_nonzeros;
		og_coo_bench_data.bytes_per_coospmv = bytes_per_coo_spmv(&coo);
	}

    /* Benchmarking */
    benchmark_coo_spmv(&coo, &x, &y, y_parallel);

    /* Test correctnesss */
	#ifdef TESTING
	coo_matrix coo_act;
	float * x_act;
	float * y_act;
	float * y_parallel_act;
	if (rank == 0){
		_rank_zero_startup(&coo_act, &x_act, &y_act, &y_parallel_act, mm_filename);
		og_nonzeros = coo_act.num_nonzeros;
		og_coo_bench_data.n_nonzeros = coo_act.num_nonzeros;
		og_coo_bench_data.bytes_per_coospmv = bytes_per_coo_spmv(&coo_act);
	}
	coo_spmv_mpi_cuda(&coo_act, &x_act, &y_act, y_parallel_act);
	if (rank == 0){
		coo_matrix coo_exp;
		read_coo_matrix(&coo_exp, mm_filename);
	
		float * x_exp = (float*)malloc(coo_exp.num_cols * sizeof(float));
		float * y_exp = (float*)calloc(coo_exp.num_rows, sizeof(float));
		init_matrix_and_xy_vals(&coo_exp, x_exp, y_exp);
		coo_spmv(&coo_exp, x_exp, y_exp);
		float max_diff = 0;
		for(int i = 0; i < coo.num_rows; i++) {
			max_diff = max(max_diff, fabs(y_parallel_act[i] - y_exp[i]));
		}
		printf("Max difference: %f\n", max_diff);
		free(y_exp);
		free(x_act);
		free(y_act);
		free(y_parallel_act);
		free(x_exp);
		delete_coo_matrix(&coo_exp);
	}
	#endif

    if (rank == 0){
		// Other nodes free their memory already after the reduce
		free(x);
		free(y);
		free(y_parallel);
	}
	delete_coo_matrix(&coo);
	if (rank == 0){
		printf("DONE\n");
	}
	MPI_Finalize();
    return 0;
}

