#include "matrix_funcs.h"

float* transpose(int N, float* src) {
	float* tar = new(std::align_val_t(64)) float[N * N];
	for (int i = 0; i < N; ++i) {
		for (int k = 0; k < N; ++k) {
			tar[i*N + k] = src[k*N + i];
		}
	}
	return tar;
}

void print_matrix(int N, int M, float* matrix){
	for (int i = 0; i < N; i++){
		for (int j = 0; j < M; j++){
			std::cout << matrix[i*M + j] << " ";
		}
		std::cout << std::endl;
	}
}