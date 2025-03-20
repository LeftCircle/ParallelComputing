#include "tests.h"





// Inside blocked_simd function:
// ...existing code...
// Replace the sequential sum with SIMD operations
// __m128 sum = _mm_setzero_ps();

// // Process left neighbors (4 at a time)
// for (int i = left_boundary; i <= right_boundary-4; i+=4) {
//     if (i != y) {
//         __m128 neighbors = _mm_load_ps(&vec[x*N + i]);  // Load 4 floats
//         sum = _mm_add_ps(sum, neighbors);               // Add in parallel
//     }
// }
// // Handle remaining elements
// for (int i = right_boundary-3; i <= right_boundary; i++) {
//     if (i != y) {
//         sum = _mm_add_ps(sum, _mm_load_ss(&vec[x*N + i]));
//     }
// }

// // Similar for vertical neighbors using transposed array
// for (int i = bottom_boundary; i <= upper_boundary-4; i+=4) {
//     if (i != x) {
//         __m128 neighbors = _mm_load_ps(&trans[y*N + i]);
//         sum = _mm_add_ps(sum, neighbors);
//     }
// }
// // Handle remaining elements
// for (int i = upper_boundary-3; i <= upper_boundary; i++) {
//     if (i != x) {
//         sum = _mm_add_ps(sum, _mm_load_ss(&trans[y*N + i]));
//     }
// }

// // Horizontal add of the 4 floats in sum vector
// float scalar_sum;
// sum = _mm_hadd_ps(sum, sum);    // [a+b,c+d,a+b,c+d]
// sum = _mm_hadd_ps(sum, sum);    // [a+b+c+d,a+b+c+d,a+b+c+d,a+b+c+d]
// _mm_store_ss(&scalar_sum, sum); // Store the final sum

// tmp[x*N + y] = scalar_sum / neighbors_count;
// // ...existing code...


void test_simd_add(){
	// Test that SIMD can add two vectors?
	const int N = 64;
	float* vec_0 = new(std::align_val_t(64)) float[N];
	float* vec_1 = new(std::align_val_t(64)) float[N];
	float* vec_sum = new(std::align_val_t(64)) float[N];
	float* expected_sum = new(std::align_val_t(64)) float[N];
	for (int i = 0; i < N; i++){
		vec_0[i] = i;
		vec_1[i] = i;
		expected_sum[i] = 2*i;
	}
	simd_vec_add_m128(N, vec_0, vec_1, vec_sum);

	assert_vectors_match(N, vec_sum, expected_sum);
	delete[] vec_0;
	delete[] vec_1;
	delete[] vec_sum;
	delete[] expected_sum;
	printf("SIMD Add Test Passed\n");
}

void test_unaligned_add(){
	// Test that SIMD can add two vectors when
	// the vector size is not divisible by 4
	const int N = 67;
	float* vec_0 = new(std::align_val_t(64)) float[N];
	float* vec_1 = new(std::align_val_t(64)) float[N];
	float* vec_sum = new(std::align_val_t(64)) float[N];
	float* expected_sum = new(std::align_val_t(64)) float[N];
	for (int i = 0; i < N; i++){
		vec_0[i] = i;
		vec_1[i] = i;
		expected_sum[i] = 2*i;
	}
	simd_vec_add_m128(N, vec_0, vec_1, vec_sum);

	assert_vectors_match(N, vec_sum, expected_sum);
	delete[] vec_0;
	delete[] vec_1;
	delete[] vec_sum;
	delete[] expected_sum;
	printf("Unaligned SIMD Add Test Passed\n");
}

// Tests a simd implementation of a 1D average stencil
// Where the stencil is a 1D average of the 8 neighbors
void test_1D_average_stencil(){
	const int N = 8;
	const int K = 2;
	float* vec = new(std::align_val_t(64)) float[N];
	//float* expected = new(std::align_val_t(64)) float[N];

	float expected_start[8] = {2, 2, 4, 4, 6, 6, 8, 8};
	for (int i = 0; i < N; i++){
		vec[i] = expected_start[i];
	}
	float expected_end[8] = {2, 3, 4, 5, 6, 7, 8, 8};

	average_1D_stencil(N, K, vec);
	for (int i = 0; i < N; i++){
		printf("%f\n", vec[i]);
	}
	assert_vectors_match(N, vec, expected_end);

	delete[] vec;
	printf("1D Average Stencil Test Passed\n");
}

void test_simd_sequential_add(){
	// Test that SIMD can average a segment of a vector
	const int N = 8;
	const int K = 2;
	float* vec = new(std::align_val_t(64)) float[N];
	float expected[8] = {2, 3, 4, 5, 6, 7, 8, 8};
	float start_vec[8] = {2, 2, 4, 4, 6, 6, 8, 8};
	for (int i = 0; i < N; i++){
		vec[i] = start_vec[i];
	}

	float seq_add = simd_sequential_add_m128(vec, 0, 4);
	assert(seq_add == 12);

	delete[] vec;
	printf("SIMD Average Block Test Passed\n");

}

void test_simd_1D_stencil(){
	// const int N = 64;
	// const int K = 8;
	// float* vec = new(std::align_val_t(64)) float[N];
	// float* expected = new(std::align_val_t(64)) float[N];
	// for (int i = 0; i < N; i++){
	// 	vec[i] = i;
	// 	expected[i] = i;
	// }
	// average_1D_stencil(N, K, expected);
	// average_1D_stencil_simd(N, K, vec);
	// assert_vectors_match(N, vec, expected);
	// delete[] vec;
	// delete[] expected;
	// printf("SIMD 1D Average Stencil Test Passed\n");
}




void run_tests(){
	test_simd_add();
	test_unaligned_add();
	test_1D_average_stencil();
	test_simd_sequential_add();
	printf("All Tests Passed\n");
}
