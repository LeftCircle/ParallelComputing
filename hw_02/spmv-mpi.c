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
#include "spmv-tests.h"

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
	// Start up MPI
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	printf("mpi node %d of %d is up and running\n", rank, size);

    if (get_arg(argc, argv, "help") != NULL){
        if (rank == 0){
			usage(argc, argv);
		}
		MPI_Finalize();
        return 0;
    }

	// Only rank 0 will read the matrix
    char * mm_filename = NULL;
    if (argc == 1) {
        printf("Give a MatrixMarket file.\n");
		MPI_Finalize();
        return -1;
    } else if (rank == 0){ 
        mm_filename = argv[1];
	}

	// The size of the arrays that node 0 will send
	coo_matrix coo;
	float * x;
	float * y;

	if (rank == 0){
    	//coo_matrix coo;
    	read_coo_matrix(&coo, mm_filename);
		x = (float*)malloc(coo.num_cols * sizeof(float));
		y = (float*)malloc(coo.num_rows * sizeof(float));
		init_matrix_and_xy_vals(&coo, x, y);
	
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

		// Now we have to distribute the data. 
		// Each node will get the entire y array, a full x array that will
		// later be summed up, and a portion of the row/column/value from the
		// coo array
		int *workload_array_size = (int *)malloc(size * sizeof(int));
		int *workload_displsi = (int*)malloc(size * sizeof(int));
		split_workload(coo.num_nonzeros, size, workload_array_size, workload_displsi);
		
		MPI_Scatter(workload_array_size, 1, MPI_INT, &coo.num_nonzeros, 1, MPI_INT,
					0, MPI_COMM_WORLD);
		
		MPI_Scatterv(coo.rows, workload_array_size, workload_displsi, MPI_INT,
					 coo.rows, coo.num_nonzeros, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Scatterv(coo.cols, workload_array_size, workload_displsi, MPI_INT,
					 coo.cols, coo.num_nonzeros, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Scatterv(coo.vals, workload_array_size, workload_displsi, MPI_FLOAT,
					 coo.vals, coo.num_nonzeros, MPI_FLOAT, 0, MPI_COMM_WORLD);
		
		printf("Rank %d got workload size %d\n", rank, coo.num_nonzeros);
		free(workload_array_size);
		free(workload_displsi);
		#ifdef DEBUG
		printf("x values on node 0 = ");
		print_vecf(x, coo.num_cols);
		#endif
	} else{
		// Recieve workload size
		MPI_Scatter(NULL, 1, MPI_INT, &coo.num_nonzeros, 1, MPI_INT,
			0, MPI_COMM_WORLD);
		// Allocate space for the values
		coo.rows = (int*)malloc(coo.num_nonzeros * sizeof(int));
		coo.cols = (int*)malloc(coo.num_nonzeros * sizeof(int));
		coo.vals = (float*)malloc(coo.num_nonzeros * sizeof(float));

		// Now receive the buffers for this nodes portion of work
		MPI_Scatterv(NULL, NULL, NULL, MPI_INT, coo.rows, coo.num_nonzeros,
					 MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Scatterv(NULL, NULL, NULL, MPI_INT, coo.cols, coo.num_nonzeros,
		 			 MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Scatterv(NULL, NULL, NULL, MPI_FLOAT, coo.vals, coo.num_nonzeros,
					 MPI_FLOAT, 0, MPI_COMM_WORLD);	 
		#ifdef DEBUG
		printf("Rank %d got workload size %d\n", rank, coo.num_nonzeros);
		#endif
	}
	// Now send the size for the arrays so that the x and y vecs can be set
	MPI_Bcast(&coo.num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&coo.num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (rank != 0){
		x = (float *)malloc(coo.num_cols * sizeof(float));
		y = (float *)malloc(coo.num_rows * sizeof(float));
		for (int i = 0; i < coo.num_rows; i++){
			y[i] = 0;
		}
	}
	// Now we can send the x array with the data in it
	MPI_Bcast(x, coo.num_cols, MPI_FLOAT, 0, MPI_COMM_WORLD);
	
	// Each node should now have it's own coo.
	// Let's try printing it
	#ifdef DEBUG
	printf("Rank %d has %d nonzeros from %f to %f\n", rank, coo.num_nonzeros, coo.vals[0], coo.vals[coo.num_nonzeros - 1]);
	printf("Rank %d has nonzeros ");
	//printf("x ranges from %f to %f\n", x[0], x[coo.num_cols - 1]);
	#endif

	// Now everyone has their own coo, x, and y
	// Let's do the spmv
	double coo_gflops;
	coo_gflops = benchmark_coo_spmv(&coo, x, y);

	// Now reduce the y arrays to the root node
	float * y_parallel = (float*)malloc(coo.num_rows * sizeof(float));
	MPI_Reduce(y, y_parallel, coo.num_rows, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

	#ifdef DEBUG
	if (rank == 0){
		printf("y vec = ");
		print_vecf(y_parallel, coo.num_rows);
	}
	#endif
   
    /* Test correctnesss */
	#ifdef TESTING
		if (rank == 0){
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
			fprintf(fp, "%f\n", y_parallel[i]);
			}
			fclose(fp);
			printf("... done!\n");
		}
	#endif
	#ifdef SEQUENTIAL_CHECK
		// Test against sequential
		if (rank == 0){
			coo_matrix seq_coo;
			read_coo_matrix(&seq_coo, mm_filename);
			float * seq_x = (float*)malloc(seq_coo.num_cols * sizeof(float));
			float * seq_y = (float*)malloc(seq_coo.num_rows * sizeof(float));
			init_matrix_and_xy_vals(&seq_coo, seq_x, seq_y);
			double coo_gflops_s = benchmark_coo_spmv(&seq_coo, seq_x, seq_y);
			test_spmv_accuracy(y_parallel, seq_y, seq_coo.num_rows, 0.01);
		}
	#endif
	

	// Now free the stuff
	free(x);
	free(y);
	delete_coo_matrix(&coo);
	MPI_Finalize();
    return 0;
}





