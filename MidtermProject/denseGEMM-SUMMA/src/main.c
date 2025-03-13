#include "summa_opts.h"
#include "utils.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tests.h"

void distribute_matrix_blocks() {
  // TODO: Implement matrix block distribution logic
  //Define the function signature and everything else
}

void summa_stationary_a(int m, int n, int k, int nprocs, int rank) {
  // TODO: Implement SUMMA algorithm with stationary A
}

void summa_stationary_b(int m, int n, int k, int nprocs, int rank) {
  // Grid setup
  // TODO: Initialize grid variables

  // Create 2D process grid
  // TODO: Create a 2D Cartesian communicator

  // Get process coordinates
  // TODO: Get the coordinates of the process in the grid

  // Create row and column communicators
  // TODO: Create row and column communicators for the grid

  // Determine local block sizes
  // TODO: Calculate the local block sizes for each process

  // Generate random matrices on root process
  // TODO: Generate random matrices A and B on the root process

  // Allocate local matrices
  // TODO: Allocate memory for local matrices

  // Distribute matrix blocks
  // TODO: Distribute matrix blocks to all processes

  // Distribute matrix blocks with proper arguments for all processes
  // TODO: Call distribute_matrix_blocks with appropriate arguments

  // SUMMA computation
  // TODO: Implement the SUMMA computation
  //You can use the function in utils.c to perform the matrix multiplication

  // Gather results
  // TODO: Gather the results from all processes

  // Verify results
  // TODO: Verify the correctness of the results
  //You can use the function in utils.c to verify the results

}

int main(int argc, char *argv[]) {
	// check to see if -t flag is passed
	if (argc > 1 && strcmp(argv[1], "-t") == 0) {
		run_tests(argc, argv);
		return 0;
	}

	// Initialize the MPI environment
	// TODO: Initialize MPI
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int grid_size = (int)sqrt(size);
	if (size != grid_size * grid_size) {
		printf("Error: Number of processes must be a perfect square\n");
		return 1;
	}

	SummaOpts opts;
	opts = parse_args(argc, argv);
	// Broadcast options to all processes
	// TODO: Broadcast the parsed options to all processes
	// Don't all processes already get the parsed options?

	// Check if matrix dimensions are compatible with grid size
	if (opts.m % grid_size != 0 || opts.n % grid_size != 0 ||
		opts.k % grid_size != 0) {
		printf("Error: Matrix dimensions must be divisible by grid size (%d)\n",
			grid_size);
		return 1;
	}

	printf("\nMatrix Dimensions:\n");
	printf("A: %d x %d\n", opts.m, opts.k);
	printf("B: %d x %d\n", opts.k, opts.n);
	printf("C: %d x %d\n", opts.m, opts.n);
	printf("Grid size: %d x %d\n", grid_size, grid_size);
	printf("Block size: %d\n", opts.block_size);
	printf("Algorithm: Stationary %c\n", opts.stationary);
	printf("Verbose: %s\n", opts.verbose ? "true" : "false");

	// Call the appropriate SUMMA function based on algorithm variant
	if (opts.stationary == 'A') {
		summa_stationary_a(opts.m, opts.n, opts.k, size, rank);
	} else if (opts.stationary == 'B') {
		summa_stationary_b(opts.m, opts.n, opts.k, size, rank);
	} else {
		printf("Error: Unknown stationary option '%c'. Use 'A' or 'B'.\n",
				opts.stationary);
		return 1;
	}

	// Finalize the MPI environment
	MPI_Finalize();

	return 0;
}