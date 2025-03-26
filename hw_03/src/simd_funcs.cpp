#include "simd_funcs.h"



void simd_vec_add_m128(int N, const float* __restrict__ vec_0, const float* __restrict__ vec_1,
	float* __restrict__ vec_sum){
	// Add two vectors using SIMD
	for (int i = 0; i < N; i+=4){
		__m128 v0 = _mm_load_ps(&vec_0[i]);
		__m128 v1 = _mm_load_ps(&vec_1[i]);
		__m128 sum = _mm_add_ps(v0, v1);
		_mm_store_ps(&vec_sum[i], sum);
	}
}

float simd_accumulate_m128_unaligned(const float* __restrict__ vec, int start, int end){
	//assert((reinterpret_cast<uintptr_t>(vec) % 64) == 0);
	int n_standard_iters = (end - start) % 4;
	int simd_end = end - n_standard_iters;
	__m128 sum = _mm_setzero_ps();
	for (int i = start; i < simd_end; i+=4){
		__m128 neighbors = _mm_loadu_ps(&vec[i]);
		sum = _mm_add_ps(sum, neighbors);
	}
	__m128 edge_case;
	load_partial(&edge_case, n_standard_iters, &vec[simd_end]);
	sum = _mm_add_ps(sum, edge_case);
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

float simd_accumulate_m128(const float* __restrict__ vec, int start, int end){
	assert((reinterpret_cast<uintptr_t>(vec) % 64) == 0);
	// Handle initial unaligned elements
	uintptr_t addr = reinterpret_cast<uintptr_t>(&vec[start]);
	int offset_to_align = (16 - (addr % 16)) / sizeof(float);
	offset_to_align = std::min(offset_to_align, end - start);
	int aligned_start = start + offset_to_align;
	
	// Knock out the unaligned start
	__m128 edge_case;
	load_partial(&edge_case, offset_to_align, &vec[start]);
	
	// Now the aligned portion
	int remaining_elements = end - aligned_start;
	int simd_iters = remaining_elements / 4;
	int simd_end = aligned_start + simd_iters * 4;
	__m128 aligned_sum = _mm_setzero_ps();
	for (int i = aligned_start; i < simd_end; i+=4){
		__m128 neighbors = _mm_load_ps(&vec[i]);
		aligned_sum = _mm_add_ps(aligned_sum, neighbors);
	}

	// Handle the end
	int n_remaining = end - simd_end;
	__m128 end_case;
	load_partial(&end_case, n_remaining, &vec[simd_end]);

	// Sum the values
	aligned_sum = _mm_add_ps(aligned_sum, end_case);
	aligned_sum = _mm_add_ps(aligned_sum, edge_case);
	float scalar_sum;
	#ifdef __mm_hadd_ps
	aligned_sum = _mm_hadd_ps(aligned_sum, aligned_sum);
	aligned_sum = _mm_hadd_ps(aligned_sum, aligned_sum);
	_mm_store_ss(&scalar_sum, aligned_sum);
	#else
	float* aligned_sum_ptr = (float*)&aligned_sum;
	scalar_sum = aligned_sum_ptr[0] + aligned_sum_ptr[1] + aligned_sum_ptr[2] + aligned_sum_ptr[3];
	#endif
	return scalar_sum;
}

float mm_sum(__m128 vec){
	#ifdef __mm_hadd_ps
	__m128 sum = _mm_hadd_ps(vec, vec);
	sum = _mm_hadd_ps(sum, sum);
	float scalar_sum;
	_mm_store_ss(&scalar_sum, sum);
	#else
	float* vec_ptr = (float*)&vec;
	float scalar_sum = vec_ptr[0] + vec_ptr[1] + vec_ptr[2] + vec_ptr[3];
	#endif
	return scalar_sum;
}

float mm_sum_partial(__m128 vec, int n_values){
	float* vec_ptr = (float*)&vec;
	float sum = 0;
	for (int i = 0; i < n_values; i++){
		sum += vec_ptr[i];
	}
	return sum;
}

float mm_sum_sequential(__m128 vec){
	float* vec_ptr = (float*)&vec;
	float sum = vec_ptr[0] + vec_ptr[1] + vec_ptr[2] + vec_ptr[3];
	return sum;
}

void print_m128(__m128 vec){
	float* vec_ptr = (float*)&vec;
	printf("%f %f %f %f\n", vec_ptr[0], vec_ptr[1], vec_ptr[2], vec_ptr[3]);
}

float sum_vec_values(const float* vec, int start, int end){
	float sum = 0;
	for (int i = start; i < end; i++){
		sum += vec[i];
	}
	return sum;
}

// Partial load. Load n elements and set the rest to 0
// You can call this function for boudary cases.
void load_partial(__m128 * xmm, int n, float const * p) {
	__m128 t1, t2;
	switch (n) {
	case 1:
		*xmm = _mm_load_ss(p); break;
	case 2:
		*xmm = _mm_castpd_ps(_mm_load_sd((double const*)p)); break;
	case 3:
		t1 = _mm_castpd_ps(_mm_load_sd((double const*)p));
		t2 = _mm_load_ss(p + 2);
		*xmm = _mm_movelh_ps(t1, t2); break;
	case 4:
		*xmm = _mm_loadu_ps(p); break;
	default:
		*xmm = _mm_setzero_ps();
	}
	return;
}