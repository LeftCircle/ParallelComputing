#ifndef TEST_FUNCS_H
#define TEST_FUNCS_H

#include <iostream>
#include <assert.h>
#include <math.h>

bool assert_vectors_match(int N, float* vec_0, float* vec_1, float epsilon = 0.0002);
void average_1D_stencil(int N, int K, float* vec);


#endif