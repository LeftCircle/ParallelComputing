#include "stencil_funcs.h"

using namespace std;

// Apply a 1D average stencil to a vector
// Where the stencil is a 1D average of the 2*K neighbors
// The neighbors are the K neighbors to the left and right
// of the current element. The current element is not included
// in the average.
void average_1D_stencil(int N, int K, float* vec){
	float* tmp = new float[N];
	for (int x = 0; x < N; x++){
		int left_boundary = max(x - K, 0);
		int right_boundary = min(x + K, K);
		float neighbor_average = get_stencil_average_from(vec, left_boundary, right_boundary, x);
		tmp[x] = neighbor_average;
	}
	for (int x = 0; x < N; x++){
		vec[x] = tmp[x];
	}
	delete[] tmp;
}

float get_stencil_average_from(float* vec, int left_boundary, int right_boundary, int x){
	float neighbors_sum = 0;
	int neighbors_count = right_boundary - left_boundary;
	for (int i = left_boundary; i <= right_boundary; i++){
		if (i != x){
			neighbors_sum += vec[i];
		}
	}
	return neighbors_sum / neighbors_count;
}