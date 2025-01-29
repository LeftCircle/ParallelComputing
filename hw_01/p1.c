// This program determines the point-to-point message latency for pairs of
// nodes. The program uses 8 nodes, and sends messages of size
// 32kb, 64kb, 128kb, 256kb, 512kb, 1mb, 2mb
// 10 messages of each size will be sent per pair of nodes, and the average
// and standard deviation will be calculated for each pair of nodes.

#include <mpi.h>
#include <sys/time.h>
#include <stdio.h>
//#include <stdlib.h>
//#include <math.h>

#define NUM_NODES 8
#define NUM_MSGS 10


void send_startup_message_between_pairs(int rank) {
	int number;
	double start, end, rtt;
	if (rank % 2 == 0) {
		start = gettimeofday();
		MPI_Send(&number, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
		MPI_Recv(&number, 1, MPI_INT, rank + 1,
				 0, MPI_COMM_WORLD, &status);
		end = gettimeofday();
		rtt = end - start;
		printf("Rank %d: startup RTT = %f seconds\n", rank, rtt);
	} else {
		MPI_Recv(&number, 1, MPI_INT, rank - 1, 0,
				 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Send(&number, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
	}
}

double ping_pong(char *msg, int msg_size, int rank, int size){
	double start, end;
	if (rank % 2 == 0) {
		start = gettimeofday();
		MPI_Send(msg, msg_size, MPI_CHAR, rank + 1, 0, MPI_COMM_WORLD);
		MPI_Recv(msg, msg_size, MPI_CHAR, rank + 1,
				 0, MPI_COMM_WORLD, &status);
		end = gettimeofday();
	} else {
		MPI_Recv(msg, msg_size, MPI_CHAR, rank - 1,
				 0, MPI_COMM_WORLD, &status);
		MPI_Send(msg, msg_size, MPI_CHAR, rank - 1, 0, MPI_COMM_WORLD);
	}
	return end - start;
}

double stddev(double *values, int n_values, double mean) {
	double sum = 0.0;
	for (int i = 0; i < n_values; i++) {
		sum += (values[i] - mean) * (values[i] - mean);
	}
	return sqrt(sum / n_values);
}

void calculate_latency(int msg_size, int num_msgs, int rank, int size) {
	char *msg = (char *)malloc(msg_size);
	double rtt, total_time = 0.0, mean, stddev, times[num_msgs];
	MPI_Status status;

	for (int i = 0; i < num_msgs; i++) {
		rtt = ping_pong(msg, msg_size, rank, size);
		times[i] = end - start;
		total_time += times[i];
	}

	mean = total_time / num_msgs;
	total_time = 0.0;

	stddev = stddev(times, num_msgs, mean);

	if (rank % 2 == 0) {
		printf("Rank %d: Message size %d bytes, Mean latency = %f seconds,
		 		Stddev = %f seconds\n", rank, msg_size, mean, stddev);
	}

	free(msg);
}

int main(int argc, char *argv[]) {
	int rank, size;
	int msg_sizes[] = {32768, 65536, 131072, 262144, 524288, 1048576, 2097152};

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (size % 2 != 0) {
		if (rank == 0) {
			fprintf(stderr, "Even number of nodes required.\n");
		}
		MPI_Abort(MPI_COMM_WORLD, 1);
		return 1;
	}

	// Send a startup message between pairs to init the connection
	send_startup_message_between_pairs(rank);

	for (int i = 0; i < sizeof(msg_sizes) / sizeof(msg_sizes[0]); i++) {
		calculate_latency(msg_sizes[i], NUM_MSGS, rank, size);
	}


	MPI_Finalize();
	return EXIT_SUCCESS;
}