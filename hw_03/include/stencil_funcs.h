#ifndef STENCIL_FUNCS_H
#define STENCIL_FUNCS_H

#include <chrono>
#include <numeric>
#include <assert.h>
#include <math.h>
#include <vector>
#include <memory>
#include <iomanip>
// contains __m128 data type
#include <xmmintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

void average_1D_stencil(int N, int K, float* vec);
float get_stencil_average_from(float* vec, int left_boundary, int right_boundary, int x);


#endif
