#include <iostream>
#include <chrono>
#include <numeric>
#include <assert.h>
#include <math.h>
#include <vector>
#include <memory>
#include <iomanip>
// contains __m128 data type
#include <xmmintrin.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "tests.h"
#include "simd_funcs.h"
#include "stencil_funcs.h"
#include "matrix_funcs.h"

using namespace std;

#ifndef __GNUC__
#define __restrict__
#endif
const float FMIN = -10;
const float FMAX = 10;



// Print the grid for debugging purposes
template <typename T>
void print_grid(int N, T vec) {
	for (int x = 0; x < N; ++x) {
		for (int y = 0; y < N; ++y) {
			cout << setprecision(2) << vec[x*N + y] << " ";
		}
		cout << endl;
	}
}

// Test if one grid equals another with a small margin of error epsilon
bool test_grids(int N, float* __restrict__ grid_1, float* __restrict__ grid_2) {
	const double epsilon = 0.0002;
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			if (!(fabs(grid_1[i*N + j] - grid_2[i*N + j]) < epsilon)) {
				cout << "ERROR: " << grid_1[i*N + j] << " != " << grid_2[i*N + j] << endl;
				return false;
			}
		}
	}
	return true;
}

// Return a random variable between fMin and fMax
float fRand(float fMin, float fMax) {
	float f = (float)rand() / RAND_MAX;
	return fMin + f * (fMax - fMin);
}

// Initialize a N * N grid with random variables coming from a seed
float* initialize_grid(int N, float fMin, float fMax) {
	float* vec = new(std::align_val_t(64)) float[N * N];
	const unsigned int seed = 42;
	srand(seed);
	for (unsigned int x = 0; x < N; ++x) {
		for (unsigned int y = 0; y < N; ++y) {
			// For better debugging
			//vec[x*N + y] = x*N + y + 100;
			vec[x*N + y] = fRand(fMin, fMax);
		}
	}
	return vec;
}

// Basic solution for the local mean kernel, allocate a tmp array of size 
// N * N for temporary write and later read backs
void basic_solution(int N, int K, float* vec) {
	stencil_2D_basic(N, K, vec);
}

// Block size is compile parameter B
/*****  Based on this code to implement your SIMD and OMP code  *****/
template<int B>
void blocked_solution(int N, int K, float* __restrict__ vec, float* __restrict__ trans) {
	stencil_2D_blocked<B>(N, K, vec, trans);
}

/*****  SIMD code  *****/
// Put the explanation of your code here

template<int B>
void blocked_simd(int N, int K, float* __restrict__ vec, float* __restrict__ trans) {
	stencil_2D_blocked_simd<B>(N, K, vec, trans);
}
/**********************/


/*****  OMP code  *****/
// Put the explanation of your code here

template<int B>
void blocked_simd_omp(int N, int K, float* __restrict__ vec, float* __restrict__ trans) {
	stencil_2D_b_simd_openmp<B>(N, K, vec, trans);
}

/**********************/


int main(int argc, char* argv[]) {

	if (argc > 1 && strcmp(argv[1], "-t") == 0) {
		run_tests();
		return 0;
	}

	// Grid size N and amount of neighbors K
	int N = 1024;
	int K = 8;
	
	// Make it possible to initialize the variables while the program is running
	if (argc == 3) {
		N = atoi(argv[1]);
		K = atoi(argv[2]);
	}

	if (argc != 1 && argc != 3) {
		cout << "Usage:" << endl;
		cout << "./program" << endl;
		cout << "./program <N> <K>" << endl;
		exit(-1);
	}

	cout << "N: " << N << ", K: " << K << ", B: " << BLOCK_SIZE << endl;

	// Run the reference solution
	// float* reference = initialize_grid(N, FMIN, FMAX);
	auto begin = chrono::high_resolution_clock::now();
	// basic_solution(N, K, reference);
	auto end = chrono::high_resolution_clock::now();
	// cout << "Reference:  " << chrono::duration_cast<chrono::milliseconds>(end-begin).count() << "ms" << endl;

	// // Run the blocked version
	// float* blocked = initialize_grid(N, FMIN, FMAX);    
	// float* blocked_transposed = transpose(N, reference);
	// begin = chrono::high_resolution_clock::now();
	// blocked_solution<BLOCK_SIZE>(N, K, blocked, blocked_transposed);
	// end = chrono::high_resolution_clock::now();
	// cout << "Blocked:    " << chrono::duration_cast<chrono::milliseconds>(end-begin).count() << "ms" << endl;
	// ::operator delete(blocked_transposed, std::align_val_t(64));
	// ::operator delete(blocked, std::align_val_t(64));

	// /***** Run SIMD version *****/
	// float* vec_blocked_vectorized = initialize_grid(N, FMIN, FMAX);
	// float* vec_blocked_vectorized_transposed = transpose(N, vec_blocked_vectorized);
	// begin = chrono::high_resolution_clock::now();
	// //blocked_simd<BLOCK_SIZE>(N, K, vec_blocked_vectorized, vec_blocked_vectorized_transposed);
	// stencil_2D_blocked_simd<BLOCK_SIZE>(N, K, vec_blocked_vectorized, vec_blocked_vectorized_transposed);
	// end = chrono::high_resolution_clock::now();
	// cout << "SIMD: " << chrono::duration_cast<chrono::milliseconds>(end-begin).count() << "ms" << endl;
	// assert(test_grids(N, reference, vec_blocked_vectorized));
	// delete[] vec_blocked_vectorized_transposed; 
	// delete[] vec_blocked_vectorized;

	/***** Run OMP version *****/
	float* vec_blocked_vectorized_multithreaded = initialize_grid(N, FMIN, FMAX);
	float* vec_blocked_vectorized_transposed_multithreaded = transpose(N, vec_blocked_vectorized_multithreaded);
	begin = chrono::high_resolution_clock::now();
	blocked_simd_omp<BLOCK_SIZE>(N, K, vec_blocked_vectorized_multithreaded, vec_blocked_vectorized_transposed_multithreaded);
	end = chrono::high_resolution_clock::now();
	cout << "SIMD+OMP: " << chrono::duration_cast<chrono::milliseconds>(end-begin).count() << "ms" << endl;
	//assert(test_grids(N, reference, vec_blocked_vectorized_multithreaded));
	delete[] vec_blocked_vectorized_transposed_multithreaded; 
	delete[] vec_blocked_vectorized_multithreaded;
	
	// Free memory
	//delete[] reference;
	//::operator delete(reference, std::align_val_t(64));

	printf("Main finished\n");
	return 0;
}