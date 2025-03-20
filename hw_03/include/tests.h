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

#ifdef _OPENMP
#include <omp.h>
#endif

#include "simd_funcs.h"
#include "test_funcs.h"
#include "stencil_funcs.h"

void run_tests();


#endif