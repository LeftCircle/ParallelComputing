#ifndef SIMD_FUNCS_H
#define SIMD_FUNCS_H

// contains __m128 data type
#include <xmmintrin.h>
#include <assert.h>
#include <iostream>


void simd_vec_add_m128(int N, const float* __restrict__ vec_0, const float* __restrict__ vec_1,
	 				   float* __restrict__ vec_sum);
float simd_accumulate_m128_unaligned(const float* __restrict__ vec, int start, int end);
float simd_accumulate_m128(const float* __restrict__ vec, int start, int end);
float mm_sum(__m128 vec);
float mm_sum_sequential(__m128 vec);
float mm_sum_partial(__m128 vec, int n_values);
void load_partial(__m128 * xmm, int n, float const * p);
void print_m128(__m128 vec);
float sum_vec_values(const float* vec, int start, int end);

#endif