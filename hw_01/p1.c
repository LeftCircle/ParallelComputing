// -----------------------------------------
// Richard Cato
// 1/30/2025
// NCSU CSC 548 Parallel Systems
// -----------------------------------------
// This program determines the point-to-point message latency for pairs of
// nodes. The program uses 8 nodes, and sends messages of size
// 32kb, 64kb, 128kb, 256kb, 512kb, 1mb, 2mb
// 10 messages of each size will be sent per pair of nodes, and the average
// and standard deviation will be calculated for each pair of nodes.

#include <mpi.h>
#include <time.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define NUM_MSGS 10
#define NUM_SIZES 16
#define STARTING_SIZE 32768

double dt(struct timespec* start, struct timespec* end){
	return (end->tv_sec - start->tv_sec) + (end->tv_nsec - start->tv_nsec) / 1E9;
}

void send_startup_message_between_pairs(int rank) {
	int number;
	double rtt;
	struct timespec start;
	struct timespec end;
	if (rank % 2 == 0) {
		clock_gettime(CLOCK_MONOTONIC, &start);
		MPI_Send(&number, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
		MPI_Recv(&number, 1, MPI_INT, rank + 1, 0,
				 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		//end = gettimeofday();
		//rtt = end - start;
		clock_gettime(CLOCK_MONOTONIC, &end);
		rtt = dt(&start, &end);
		printf("Rank %d: startup RTT = %f seconds\n", rank, rtt);
	} else {
		MPI_Recv(&number, 1, MPI_INT, rank - 1, 0,
				 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Send(&number, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD);
	}
}

double ping_pong(char *msg, int msg_size, int rank, int size){
	struct timespec start;
	struct timespec end;
	MPI_Status status;
	if (rank % 2 == 0) {
		clock_gettime(CLOCK_MONOTONIC, &start);
		MPI_Send(msg, msg_size, MPI_CHAR, rank + 1, 0, MPI_COMM_WORLD);
		MPI_Recv(msg, msg_size, MPI_CHAR, rank + 1,
				 0, MPI_COMM_WORLD, &status);
		clock_gettime(CLOCK_MONOTONIC, &end);
	} else {
		MPI_Recv(msg, msg_size, MPI_CHAR, rank - 1,
				 0, MPI_COMM_WORLD, &status);
		MPI_Send(msg, msg_size, MPI_CHAR, rank - 1, 0, MPI_COMM_WORLD);
		return 0;
	}
	double duration = dt(&start, &end);
	//printf("dt = %f ms\n", duration * 1E3);
	return duration;
}

double standard_dev(double *values, int n_values, double mean) {
	double sum = 0.0;
	for (int i = 0; i < n_values; i++) {
		sum += (values[i] - mean) * (values[i] - mean);
	}
	return sqrt(sum / n_values);
}

void calculate_latency(int msg_size, int num_msgs, int rank, int size) {

	char *msg = (char *)malloc(msg_size);
	double rtt_0 = ping_pong(msg, msg_size, rank, size);
	if (rank % 2 == 0){
		printf("Rank %d: First message to init connection/buffers had an rtt of %f msec\n", rank, 1E3 * rtt_0);
	}

	double rtt, total_time = 0.0, mean, stddev, times[num_msgs];
	for (int i = 0; i < num_msgs; i++) {
		rtt = ping_pong(msg, msg_size, rank, size);
		times[i] = rtt;
		total_time += rtt;
	}

	mean = total_time / num_msgs;
	total_time = 0.0;

	stddev = standard_dev(times, num_msgs, mean);

	if (rank % 2 == 0) {
		//printf("Rank %d: Message size %d bytes, Mean latency = %f seconds, Stddev = %f seconds\n", rank, msg_size, mean, stddev);
		printf("Rank %d: Message size %d bytes, Mean latency = %f ms, Stddev = %f ms\n", rank, msg_size, mean * 1E3, stddev * 1E3);
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

	//for (int i = 0; i < sizeof(msg_sizes) / sizeof(msg_sizes[0]); i++) {
	//	calculate_latency(msg_sizes[i], NUM_MSGS, rank, size);
	//}

	int msg_size = STARTING_SIZE;
	for (int i = 0; i < NUM_SIZES; i++){
		msg_size = STARTING_SIZE * pow(2, i);
		calculate_latency(msg_size, NUM_MSGS, rank, size);
	}

	MPI_Finalize();
	return 0;
}