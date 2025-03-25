#ifndef STENCIL_FUNCS_H
#define STENCIL_FUNCS_H

#include <chrono>
#include <numeric>
#include <assert.h>
#include <math.h>
#include <vector>
#include <memory>
#include <iomanip>
// contains __m128 data type
#include <xmmintrin.h>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "simd_funcs.h"

const int BLOCK_SIZE = 2;

void average_1D_stencil(int N, int K, float* vec);
float get_stencil_average_from(const float* vec, int left_boundary, int right_boundary, int x);
float get_stencil_average_from_mm128(const float* vec, int left_boundary, int right_boundary, int x);

void stencil_2D_basic(int N, int K, float* vec);

template<int B>
void stencil_2D_blocked(int N, int K, float* __restrict__ vec, float* __restrict__ trans);
template<int B>
void stencil_2D_blocked_simd(int N, int K, float* __restrict__ vec, const float* __restrict__ trans);


void _average_1D_stencil(int N, int K, const float* vec, float* tmp);
void _average_2D_stencil(int N, int K, int start_x, int start_y, int block_size, const float* __restrict__ vec,
	 					const float* __restrict__ trans, float* __restrict__ out);
// void _average_1D_stencil_mm128(int N, int K, const float* vec, float* tmp);

#endif
