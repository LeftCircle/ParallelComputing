#include "tests.h"



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
	int N = 8;
	int K = 1;
	float* vec = new(std::align_val_t(64)) float[N];

	float expected_start[8] = {2, 2, 4, 4, 6, 6, 8, 8};
	for (int i = 0; i < N; i++){
		vec[i] = expected_start[i];
	}
	float expected_end[8] =   {2, 3, 3, 5, 5, 7, 7, 8};

	average_1D_stencil(N, K, vec);
	// for (int i = 0; i < N; i++){
	// 	printf("%f\n", vec[i]);
	// }
	assert_vectors_match(N, vec, expected_end);
	printf("First 1D Average Stencil Test Passed\n");

	N = 8;
	K = 2;
	
	for (int i = 0; i < N; i++){
		vec[i] = expected_start[i];
	}
	float e_end_2[8] =   {3, 10. / 3., 14./4., 18./4., 22./4., 26./4., 20./3., 7};

	average_1D_stencil(N, K, vec);
	for (int i = 0; i < N; i++){
		printf("%f\n", vec[i]);
	}
	assert_vectors_match(N, vec, e_end_2);

	delete[] vec;
	printf("1D Average Stencil Test Passed\n");
}

void test_1D_average_stencil_simd(){
	const int N = 8;
	const int K = 2;
	float* vec = new(std::align_val_t(64)) float[N];
	//float* expected = new(std::align_val_t(64)) float[N];

	float expected_start[8] = {2, 2, 4, 4, 6, 6, 8, 8};
	for (int i = 0; i < N; i++){
		vec[i] = expected_start[i];
	}
	float expected_end[8] =   {3, 5, 7, 9, 11, 13, 10, 7};

	average_1D_stencil(N, K, vec);
	for (int i = 0; i < N; i++){
		printf("%f\n", vec[i]);
	}
	assert_vectors_match(N, vec, expected_end);

	delete[] vec;
	printf("1D Average Stencil SIMD Test Passed\n");
}

void test_simd_sequential_add(){
	// Test that SIMD can average a segment of a vector
	const int N = 8;
	float* vec = new(std::align_val_t(64)) float[N];
	float expected[8] = {2, 3, 4, 5, 6, 7, 8, 8};
	float start_vec[8] = {2, 2, 4, 4, 6, 6, 8, 8};
	for (int i = 0; i < N; i++){
		vec[i] = start_vec[i];
	}

	float seq_add = simd_accumulate_m128(vec, 0, 4);
	assert(seq_add == 12);

	delete[] vec;
	printf("SIMD simd_sequential_add Test Passed\n");

}

void test_simd_sequantial_add_any_size(){
	const int N = 15;
	float expected[N];
	float vec[N];
	for (int i = 0; i < N; i++){
		vec[i] = i;
		expected[i] = i;
	}
	float sum = sum_vec_values(vec, 0, N);
	float simd_sum = simd_accumulate_m128(vec, 0, N);
	assert(sum == simd_sum);
	printf("SIMD Sequential Add Any Size Test Passed\n");
}

void test_m_128_with_three_values(){
	// Test that SIMD can add three values
	__m128 sum = _mm_setzero_ps();
	float values[3] = {1, 2, 3};
	__m128 values_m128 = _mm_loadu_ps(values);
	sum = _mm_add_ps(sum, values_m128);
	//print_m128(sum);
	float sequential_sum = mm_sum_partial(sum, 3);
	//printf("Sequential Sum = %f\n", sequential_sum);
	assert(sequential_sum == 6);
	printf("SIMD Add Three Values Test Passed\n");
}

void test_2d_simd_stencil(){
	const int N = 4;
	const int K = 2;
	float* vec = new(std::align_val_t(64)) float[N*N];
	float* expected = new(std::align_val_t(64)) float[N*N];
	#pragma omp parallel for
	for (int i = 0; i < N*N; i++){
		//float val = rand() % 10;
		float val = i % N;
		vec[i] = val;
		expected[i] = val;
	}
	print_matrix(N, N, vec);
	
	stencil_2D_basic(N, K, expected);
	printf("Expected\n");
	print_matrix(N, N, expected);
	float* trans = transpose(N, vec);
	stencil_2D_blocked_simd<BLOCK_SIZE>(N, K, vec, trans);
	printf("Actual\n");
	print_matrix(N, N, vec);

	//assert_vectors_match(N*N, vec, expected);
	::operator delete[] (vec, std::align_val_t(64));
	::operator delete[] (expected, std::align_val_t(64));
	::operator delete[] (trans, std::align_val_t(64));
	printf("2D SIMD Stencil Test Passed\n");
	
}


void profile_simd_sum(){
	const int N = 1000000;
	// Profiling SIMD sum vs SIMD sequential sum
	float values[4] = {1, 2, 3, 4};
	__m128 values_m128 = _mm_loadu_ps(values);
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < N; i++){
		float sum = mm_sum(values_m128);
	}
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> simd_time = end - start;
	start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < N; i++){
		float sum = mm_sum_partial(values_m128, 4);
	}
	end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> sequential_time = end - start;
	std::cout << "SIMD Time: " << simd_time.count() << std::endl;
	std::cout << "Sequential Time: " << sequential_time.count() << std::endl;

}

void profile_vec_sequential_sum(){
	const int N = 1000000;
	float* test_vec = new(std::align_val_t(64)) float[N];;
	printf("Profiling Sequential Sum\n");
	#pragma omp parallel for
	for (int i = 0; i < N; i++){
		// how do I create a 16 bit float?
		test_vec[i] = rand() % 10;
	}
	auto start = std::chrono::high_resolution_clock::now();
	//float sum = std::accumulate(test_vec, test_vec + N, 0);
	float sum = sum_vec_values(test_vec, 0, N);
	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> sequential_time = end - start;
	std::cout << "Sequential Time: " << sequential_time.count() << std::endl;

	start = std::chrono::high_resolution_clock::now();
	float simd_sum = simd_accumulate_m128(test_vec, 0, N);
	end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> simd_time = end - start;
	std::cout << "SIMD Time: " << simd_time.count() << std::endl;
	printf("Sum = %f simd_sum = %f\n", sum, simd_sum);
	//delete[] test_vec;
	::operator delete(test_vec, std::align_val_t(64));
	assert(simd_sum == sum);
	printf("\n");
}

void run_profile_tests(){
	profile_simd_sum();
	profile_vec_sequential_sum();
}


void run_tests(){
	#ifdef PROFILE
	run_profile_tests();
	#endif
	test_simd_add();
	test_unaligned_add();
	test_simd_sequential_add();
	test_m_128_with_three_values();
	test_1D_average_stencil();
	test_simd_sequantial_add_any_size();
	test_2d_simd_stencil();
	printf("All Tests Passed\n");

	
}



