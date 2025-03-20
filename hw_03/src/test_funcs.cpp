#include "test_funcs.h"


bool assert_vectors_match(int N, float* vec_0, float* vec_1, float epsilon){
	for (int i = 0; i < N; ++i) {
		if (!(fabs(vec_0[i] - vec_1[i]) < epsilon)) {
			printf("ERROR: %f != %f\n", vec_0[i], vec_1[i]);
			assert(false);
			return false;
		}
	}
	return true;
}