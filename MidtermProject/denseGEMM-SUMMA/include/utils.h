// utils.h
#ifndef __UTILS_H__
#define __UTILS_H__

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "summa_opts.h"

void matmul(float* A, float* B, float* C, int m, int n, int k);
void verify_result(float* C_global, float* A, float* B, int m, int n, int k);
float* generate_matrix_A(int rows, int cols, int rank);
float* generate_matrix_B(int rows, int cols, int rank);
float* generate_matrix(int rows, int cols);
float* generate_int_matrix(int rows, int cols, int rank);

//float* generate_int_matrix(int rows, int cols, int rank);
//Matrix* init_c_matrix_for_stationary_c(int m, int k, int n, int n_processors, int rank);
//void init_a_matrix_for_stationary_c_summa(Matrix* A, int m, int k, int n_processors, int rank);

#endif