#include "stencil_funcs.h"

using namespace std;

// Apply a 1D average stencil to a vector
// Where the stencil is a 1D average of the 2*K neighbors
// The neighbors are the K neighbors to the left and right
// of the current element. The current element is not included
// in the average.
void average_1D_stencil(int N, int K, float* vec){
	float* tmp = new float[N];
	_average_1D_stencil(N, K, vec, tmp);
	memcpy(vec, tmp, N * sizeof(float));
	delete[] tmp;
}

float get_stencil_average_from(const float* vec, int left_boundary, int right_boundary, int x){
	// Figure out the neighbors in left boundary
	int l_neighbors = x - left_boundary;
	int r_neighbors = right_boundary - x;
	// Now sum the values of the neighbors
	float left_sum = sum_vec_values(vec, left_boundary, x);
	float right_sum = sum_vec_values(vec, x + 1, right_boundary + 1);
	//printf("Left Sum = %f Right Sum = %f\n", left_sum, right_sum);
	return (left_sum + right_sum) / (l_neighbors + r_neighbors);
}

float get_stencil_average_from_mm128(const float* vec, int left_boundary, int right_boundary, int x){
	// Figure out the neighbors in left boundary
	int l_neighbors = x - left_boundary;
	int r_neighbors = right_boundary - x;
	// Now sum the values of the neighbors
	float left_sum = simd_accumulate_m128(vec, left_boundary, x);
	float right_sum = simd_accumulate_m128(vec, x + 1, right_boundary + 1);
	return (left_sum + right_sum) / (l_neighbors + r_neighbors);
}

void _average_1D_stencil(int N, int K, const float* vec, float* tmp){
	for (int x = 0; x < N; x++){
		int left_boundary = max(x - K, 0);
		int right_boundary = min(x + K, N - 1);
		float neighbor_average = get_stencil_average_from_mm128(vec, left_boundary, right_boundary, x);
		tmp[x] = neighbor_average;
	}
}

// Solutions:
template<int B>
void stencil_2D_blocked(int N, int K, float* __restrict__ vec, float* __restrict__ trans){
	// Divisibility contraints
	if (N % B != 0) {
		cout << "N must be divisible through B" << endl;
		exit(-1);
	}
	// Allocate tmp grid
	float* tmp = new(std::align_val_t(64)) float[N*N];

	// Find neighbors
	for (int I = 0; I < N; I+=B) {
		for (int J = 0; J < N; J+=B) {
			for (int x = I; x < I + B; ++x) {
				for (int y = J; y < J + B; ++y) {
					int left_boundary = max(y - K, 0), right_boundary = min(y + K, N - 1);
					int bottom_boundary = max(x - K, 0), upper_boundary = min(x + K, N - 1);
					int neighbors_count = (right_boundary - left_boundary) + (upper_boundary - bottom_boundary);
					float sum = 0.0;

					for (int i = left_boundary; i <= right_boundary; i++) {
						if (i != y) {
							sum += vec[x*N + i];
						}
					}

					for (int i = bottom_boundary; i <= upper_boundary; i++) {
						if (i != x) {
							sum += trans[y*N+i]; // Makes it a lot faster due to consecutive reads
						}
					}

					// Replace current value at u(x, y) with local mean
					tmp[x*N + y] = sum / neighbors_count;
				}
			}
		}
	}

	// Write back
	for (int x = 0; x < N; ++x) {
		for (int y = 0; y < N; ++y) {
			vec[x*N + y] = tmp[x*N + y];
		}
	}
	delete[] tmp;
}

void stencil_2D_basic(int N, int K, float* vec){
	// Allocate tmp grid
	float* tmp = new(std::align_val_t(64)) float[N*N];

	// Find neighbors
	for (int x = 0; x < N; x++) {
		for (int y = 0; y < N; y++) {
			int left_boundary = max(y - K, 0), right_boundary = min(y + K, N - 1);
			int bottom_boundary = max(x - K, 0), upper_boundary = min(x + K, N - 1);
			int neighbors_count = (upper_boundary - bottom_boundary) + (right_boundary - left_boundary);
			float sum = 0.0;

			for (int i = left_boundary; i <= right_boundary; i++) {
				if (i != y) {
					sum += vec[x*N + i];
				}
			}

			for (int i = bottom_boundary; i <= upper_boundary; i++) {
				if (i != x) {
					sum += vec[i*N + y];
				}
			}

			// Replace current value at u(x, y) with local mean
			tmp[x*N + y] = sum / neighbors_count;
		}
	}

	// Write back
	for (int x = 0; x < N; ++x) {
		for (int y = 0; y < N; ++y) {
			vec[x*N + y] = tmp[x*N + y];
		}
	}
	delete[] tmp;
}

void stencil_2D_simd(int N, int K, float* __restrict__ vec, const float* __restrict__ trans){
	float* tmp = new(std::align_val_t(64)) float[N*N];

	// Find neighbors
	for (int x = 0; x < N; x++) {
		for (int y = 0; y < N; y++) {
			_average_2D_stencil_point(N, K, x, y, vec, trans, tmp);
		}
	}
	memcpy(vec, tmp, N*N * sizeof(float));
	operator delete (tmp, std::align_val_t(64));
}

template<int B>
void stencil_2D_blocked_simd(int N, int K, float* __restrict__ vec, const float* __restrict__ trans){
	if (N % B != 0) {
		cout << "N must be divisible through B" << endl;
		exit(-1);
	}
	// Allocate tmp grid
	float* tmp = new(std::align_val_t(64)) float[N*N];
	for (int I = 0; I < N; I+=B) {
		for (int J = 0; J < N; J+=B) {
			_average_2D_stencil_blocked<B>(N, K, I, J, vec, trans, tmp);
		}
	}

	// Write back
	memcpy(vec, tmp, N*N * sizeof(float));
	::operator delete(tmp, std::align_val_t(64));
}

template<int B>
void _average_2D_stencil_blocked(int N, int K, int start_x, int start_y, const float* __restrict__ vec,
								 const float* __restrict__ trans, float* __restrict__ out){
	int stop_x = min(start_x + B, N);
	int stop_y = min(start_y + B, N);
	for (int x = start_x; x < stop_x; x++){
		for (int y = start_y; y < stop_y; y++){
			_average_2D_stencil_point(N, K, x, y, vec, trans, out);
		}
	}
}

template<int B>
void stencil_2D_b_simd_openmp(int N, int K, float* __restrict__ vec, const float* __restrict__ trans){
	if (N % B != 0) {
		cout << "N must be divisible through B" << endl;
		exit(-1);
	}
	// Allocate tmp grid
	float* tmp = new(std::align_val_t(64)) float[N*N];
	#pragma omp parallel for
	for (int I = 0; I < N; I+=B) {
		for (int J = 0; J < N; J+=B) {
			_average_2D_stencil_blocked<B>(N, K, I, J, vec, trans, tmp);
		}
	}

	// Write back
	memcpy(vec, tmp, N*N * sizeof(float));
	delete[] tmp;
}

void _average_2D_stencil_point(int N, int K, int x, int y, const float* __restrict__ vec,
	const float* __restrict__ trans, float* __restrict__ out){
	int left_boundary = max(y - K, 0);
	int right_boundary = min(y + K, N - 1);
	int bottom_boundary = max(x - K, 0);
	int upper_boundary = min(x + K, N - 1);
	int l_neighbors = y - left_boundary;
	int r_neighbors = right_boundary - y;
	int b_neighbors = x - bottom_boundary;
	int u_neighbors = upper_boundary - x;
	int neighbors_count = l_neighbors + r_neighbors + b_neighbors + u_neighbors;
	float l_sum = simd_accumulate_m128_unaligned(vec, x * N + left_boundary, x * N + y);
	float r_sum = simd_accumulate_m128_unaligned(vec, x * N + y + 1, x * N + right_boundary + 1);
	float b_sum = simd_accumulate_m128_unaligned(trans, y * N + bottom_boundary, y * N + x);
	float t_sum = simd_accumulate_m128_unaligned(trans, y * N + x + 1, y * N + upper_boundary + 1);
	float sum = l_sum + r_sum + b_sum + t_sum;
	out[x*N + y] = sum / neighbors_count;
}


// Declare the blocked solution
template void stencil_2D_blocked<BLOCK_SIZE>(int N, int K, float* __restrict__ vec, float* __restrict__ trans);

template void stencil_2D_blocked_simd<BLOCK_SIZE>(int N, int K, float* __restrict__ vec,
												 const float* __restrict__ trans);

template void _average_2D_stencil_blocked<BLOCK_SIZE>(int N, int K, int start_x,
													  int start_y, const float* __restrict__ vec,
													  const float* __restrict__ trans, float* __restrict__ out);
template void stencil_2D_b_simd_openmp<BLOCK_SIZE>(int N, int K, float* __restrict__ vec,
												   const float* __restrict__ trans);