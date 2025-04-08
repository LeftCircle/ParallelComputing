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

//#include <omp.h>


#define PROFILE

void run_tests();
void run_profile_tests();

#endif