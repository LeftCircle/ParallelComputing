#include "simd_funcs.h"



void simd_vec_add_m128(int N, float* __restrict__ vec_0, float* __restrict__ vec_1, float* __restrict__ vec_sum){
	// Add two vectors using SIMD
	for (int i = 0; i < N; i+=4){
		__m128 v0 = _mm_load_ps(&vec_0[i]);
		__m128 v1 = _mm_load_ps(&vec_1[i]);
		__m128 sum = _mm_add_ps(v0, v1);
		_mm_store_ps(&vec_sum[i], sum);
	}
}

float simd_sequential_add_m128(float* __restrict__ vec, int start, int end){
	assert((end - start) % 4 == 0);
	__m128 sum = _mm_setzero_ps();
	for (int i = start; i < end; i+=4){
		__m128 neighbors = _mm_load_ps(&vec[i]);
		sum = _mm_add_ps(sum, neighbors);
	}
	float scalar_sum;
	#ifdef __mm_hadd_ps
	sum = _mm_hadd_ps(sum, sum);
	sum = _mm_hadd_ps(sum, sum);
	_mm_store_ss(&scalar_sum, sum);
	#else
	float* sum_ptr = (float*)&sum;
	scalar_sum = sum_ptr[0] + sum_ptr[1] + sum_ptr[2] + sum_ptr[3];
	#endif
	return scalar_sum;
}