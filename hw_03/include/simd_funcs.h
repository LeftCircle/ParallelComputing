#ifndef SIMD_FUNCS_H
#define SIMD_FUNCS_H

// contains __m128 data type
#include <xmmintrin.h>
#include <assert.h>

const int BLOCK_SIZE = 32;


void simd_vec_add_m128(int N, float* __restrict__ vec_0, float* __restrict__ vec_1, float* __restrict__ vec_sum);
float simd_sequential_add_m128(float* __restrict__ vec, int start, int end);


#endif