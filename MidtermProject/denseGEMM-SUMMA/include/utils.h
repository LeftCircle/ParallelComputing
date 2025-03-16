// utils.h
#ifndef __UTILS_H__
#define __UTILS_H__

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#include "summa_opts.h"

typedef struct{
	MPI_Comm parent_comm;
	MPI_Comm cart_comm;
}CartCommunicator;

void matmul(float* A, float* B, float* C, int m, int n, int k);
void verify_result(float* C_global, float* A, float* B, int m, int n, int k);
float* generate_matrix_A(int rows, int cols, int rank);
float* generate_matrix_B(int rows, int cols, int rank);
float* generate_matrix(int rows, int cols);
float* generate_int_matrix(int rows, int cols, int rank);
float* generate_rank_matrix(int rows, int cols, int rank);
bool do_matrices_match(float* A, float* B, int rows, int cols, float tolerance);

void scatter_row_major_matrix(float* global_matrix, float* local_matrix, int m, int k,
	int grid_size, int rank, int size, MPI_Comm comm);

MPI_Datatype create_block_type(int m, int k, int grid_size);
CartCommunicator create_cartesian_topology(MPI_Comm comm, int grid_size);
float* scatter_matrix(float* matrix, int rank, int size, int m, int k, MPI_Comm comm);
float* init_c_matrix_for_stationary_c(int m, int k, int n, int n_processors, int rank);
void gather_row_major_matrix(float* local_matrix, float* global_matrix, 
						int m, int n, int grid_size, int rank, int size, MPI_Comm comm);

void gather_col_blocks_into_root_matrix(float* local_matrix, float* global_matrix,
					 int m, int n, int grid_size, int rank, int size, int col,
					 MPI_Comm cart_comm, MPI_Comm row_comm);

void set_send_offset_for_block_scat_gath(int* sendcounts, int* displs, int m,
										int k, int grid_size, MPI_Comm comm);
void set_send_offset_for_row_block_gatherv(int* sendcounts, int* displs, int p_col,
								int m, int n, int grid_size, MPI_Comm comm);
void set_send_offset_for_col_block_gatherv(int* sendcounts, int* displs, int p_col,
								int m, int n, int grid_size, MPI_Comm comm);

void print_matrix(float* matrix, int rows, int cols);

float* stationary_c_summa(int m, int k, int n, int rank, int size);
void place_submatrix_into_full_matrix(float* full_matrix, float* sub_matrix, int m, int n, int local_m, int local_n, int* sub_position);

bool is_in_array(int* array, int size, int value);

void broadcast_matrix_to_column(float* send_buff, float* rcv_buff, int count, int from_rank,
									 int to_col, int grid_size, int rank, MPI_Comm comm);

//float* generate_int_matrix(int rows, int cols, int rank);
//Matrix* init_c_matrix_for_stationary_c(int m, int k, int n, int n_processors, int rank);
//void init_a_matrix_for_stationary_c_summa(Matrix* A, int m, int k, int n_processors, int rank);

#endif