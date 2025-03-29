
#include <stdio.h>
#include <cuda_runtime.h>

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

void usage(int argc, char** argv)
{
    printf("Usage: %s [my_matrix.mtx]\n", argv[0]);
    printf("Note: my_matrix.mtx must be real-valued sparse matrix in the MatrixMarket file format.\n"); 
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
}

double time_coo_smpv_kernal(coo_matrix* coo, int* d_rows, int* d_cols, float* d_vals, float* d_x, float* d_y) {
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
	double msec_per_iteration = milliseconds_elapsed(&t) / (double) num_iterations;
    double sec_per_iteration = msec_per_iteration / 1000.0;
    double GFLOPs = (sec_per_iteration == 0) ? 0 : (2.0 * (double) coo->num_nonzeros / sec_per_iteration) / 1e9;
    double GBYTEs = (sec_per_iteration == 0) ? 0 : ((double) bytes_per_coo_spmv(coo) / sec_per_iteration) / 1e9;
    printf("\tbenchmarking COO-SpMV: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s)\n", msec_per_iteration, GFLOPs, GBYTEs);
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

double benchmark_coo_spmv(coo_matrix * coo, float* x, float* y){
	// Device memory pointers
	int *d_rows, *d_cols;
	float *d_vals, *d_x, *d_y;
	send_coo_spmv_data_to_gpu(coo, x, &d_rows, &d_cols, &d_vals, &d_x, &d_y);
	
	double time = time_coo_smpv_kernal(coo, d_rows, d_cols, d_vals, d_x, d_y);

	// Copy result back to host
	cudaMemcpy(y, d_y, coo->num_rows * sizeof(float), cudaMemcpyDeviceToHost);
	
	// Cleanup
	cudaFree(d_rows);
	cudaFree(d_cols);
	cudaFree(d_vals);
	cudaFree(d_x);
	cudaFree(d_y);
	return time;
}

int main(int argc, char** argv)
{
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
    read_coo_matrix(&coo, mm_filename);

    // fill matrix with random values: some matrices have extreme values, 
    // which makes correctness testing difficult, especially in single precision
    srand(13);
    for(int i = 0; i < coo.num_nonzeros; i++) {
        coo.vals[i] = 1.0 - 2.0 * (rand() / (RAND_MAX + 1.0)); 
	}
    
    printf("\nfile=%s rows=%d cols=%d nonzeros=%d\n", mm_filename, coo.num_rows, coo.num_cols, coo.num_nonzeros);
    fflush(stdout);

#ifdef TESTING
//print in COO format
    printf("Writing matrix in COO format to test_COO ...");
    FILE *fp = fopen("test_COO", "w");
    fprintf(fp, "%d\t%d\t%d\n", coo.num_rows, coo.num_cols, coo.num_nonzeros);
    fprintf(fp, "coo.rows:\n");
    for (int i=0; i<coo.num_nonzeros; i++)
    {
      fprintf(fp, "%d  ", coo.rows[i]);
    }
    fprintf(fp, "\n\n");
    fprintf(fp, "coo.cols:\n");
    for (int i=0; i<coo.num_nonzeros; i++)
    {
      fprintf(fp, "%d  ", coo.cols[i]);
    }
    fprintf(fp, "\n\n");
    fprintf(fp, "coo.vals:\n");
    for (int i=0; i<coo.num_nonzeros; i++)
    {
      fprintf(fp, "%f  ", coo.vals[i]);
    }
    fprintf(fp, "\n");
    fclose(fp);
    printf("... done!\n");
#endif 

    //initialize host arrays
    float * x = (float*)malloc(coo.num_cols * sizeof(float));
	float * y = (float*)malloc(coo.num_rows * sizeof(float));

    for(int i = 0; i < coo.num_cols; i++) {
        x[i] = rand() / (RAND_MAX + 1.0); 
    }
    for(int i = 0; i < coo.num_rows; i++)
        y[i] = 0;

    /* Benchmarking */
    double coo_gflops;
    coo_gflops = benchmark_coo_spmv(&coo, x, y);
	printf("COO GFLOPS: %f\n", coo_gflops);

    /* Test correctnesss */
	#ifdef TESTING
	float* y_act = (float*)calloc(coo.num_rows, sizeof(float));
	coo_spmv_cuda(&coo, x, y_act);
	float * y_exp = (float*)calloc(coo.num_rows, sizeof(float));
	coo_spmv(&coo, x, y_exp);
	float max_diff = 0;
	for(int i = 0; i < coo.num_rows; i++) {
		max_diff = max(max_diff, fabs(y_act[i] - y_exp[i]));
	}
	printf("Max difference: %f\n", max_diff);
	free(y_exp);
	free(y_act);

	#endif

    delete_coo_matrix(&coo);
    free(x);
    free(y);

    return 0;
}

