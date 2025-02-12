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
//    printf("estimated time for once %f\n", (float) estimated_time);

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
            y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
        }
    double msec_per_iteration = milliseconds_elapsed(&t) / (double) num_iterations;
    double sec_per_iteration = msec_per_iteration / 1000.0;
    double GFLOPs = (sec_per_iteration == 0) ? 0 : (2.0 * (double) coo->num_nonzeros / sec_per_iteration) / 1e9;
    double GBYTEs = (sec_per_iteration == 0) ? 0 : ((double) bytes_per_coo_spmv(coo) / sec_per_iteration) / 1e9;
    printf("\tbenchmarking COO-SpMV: %8.4f ms ( %5.2f GFLOP/s %5.1f GB/s)\n", msec_per_iteration, GFLOPs, GBYTEs); 

    return msec_per_iteration;
}

double benchmark_coo_spmv_mpi(coo_matrix * coo, float* x, float* y)
{
	int num_nonzeros = coo->num_nonzeros;

	// warmup    
	timer time_one_iteration;
	timer_start(&time_one_iteration);
	for (int i = 0; i < num_nonzeros; i++){   
		y[coo->rows[i]] += coo->vals[i] * x[coo->cols[i]];
	}

	double estimated_time = seconds_elapsed(&time_one_iteration);
}

int main(int argc, char** argv)
{
	// Start up MPI
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);


    if (get_arg(argc, argv, "help") != NULL){
        if (rank == 0){
			usage(argc, argv);
		}
        return 0;
    }

	// Only rank 0 will read the matrix
    char * mm_filename = NULL;
    if (argc == 1) {
        printf("Give a MatrixMarket file.\n");
        return -1;
    } else if (rank == 0){ 
        mm_filename = argv[1];
	}
	if (rank == 0){
    	coo_matrix coo;
    	read_coo_matrix(&coo, mm_filename);
	
    	// fill matrix with random values: some matrices have extreme values, 
    	// which makes correctness testing difficult, especially in single precision
    	srand(13);
		for(int i = 0; i < coo.num_nonzeros; i++) {
			coo.vals[i] = 1.0 - 2.0 * (rand() / (RAND_MAX + 1.0)); 
			// coo.vals[i] = 1.0;
		}
		
		printf("\nfile=%s rows=%d cols=%d nonzeros=%d\n", mm_filename, coo.num_rows, coo.num_cols, coo.num_nonzeros);
		fflush(stdout);
	}

#ifdef TESTING
//print in COO format
	if (rank == 0){
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
	}
#endif 

	// Start by just scattering the data then printing it
	// rank 0 needs to let each node know how much data they will get
	if (rank == 0){
		float *all_x = (int *)malloc(coo.num_cols * sizeof(int));
		float *all_y = (int *)malloc(coo.num_rows * sizeof(int));
		for (int i = 0; i < coo.num_cols; i++){
			all_x[i] = rand() / (RAND_MAX + 1.0);
		}
		for (int i = 0; i < coo.num_rows; i++){
			all_y[i] = 0;
		}

		// Now we have to distribute the data. 
		// Each node will get the entire y array, a full x array that will
		// later be summed up, and a portion of the row/column/value from the
		// coo array
		int *workload_array = (int *)malloc(size * sizeof(int));
		split_workload(coo.num_nonzeros, size, workload_array);
		
		// now we have to scatter the workload array so each node can create a
		// buffer for the correct amount of data
		int workload_size = workload_array[rank];
		MPI_Scatter(workload_array, 1, MPI_INT, &workload_size, 1, MPI_INT,
			 0, MPI_COMM_WORLD);
		
		// Now free the stuff
		free(workload_array);
		free(all_x);
		free(all_y);
		delete_coo_matrix(&coo);
	} else {
		int workload_size;
		MPI_Scatter(NULL, 1, MPI_INT, &workload_size, 1, MPI_INT,
			 0, MPI_COMM_WORLD);
		#ifdef DEBUG
		printf("Rank %d got workload size %d\n", rank, workload_size);
		#endif
	}

    /* Benchmarking */
    //double coo_gflops;
    //coo_gflops = benchmark_coo_spmv(&coo, x, y);

    /* Test correctnesss */
#ifdef TESTING
    printf("Writing x and y vectors ...");
    fp = fopen("test_x", "w");
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

    //delete_coo_matrix(&coo);
    //free(x);
    //free(y);

    return 0;
}





