// A hellow world open mpi program
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
	// Initialize the MPI environment
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	
	printf("Hello world from processor %d, rank %d \n",
		  rank, size);

	// Finalize the MPI environment
	MPI_Finalize();
}