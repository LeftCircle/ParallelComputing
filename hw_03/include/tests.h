#ifndef TESTS_H
#define TESTS_H

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

#include <omp.h>


#include "simd_funcs.h"
#include "test_funcs.h"
#include "stencil_funcs.h"
#include "matrix_funcs.h"

#define PROFILE

void run_tests();
void run_profile_tests();


#endif