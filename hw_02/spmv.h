#pragma once

#include <stdio.h>

#include "cmdline.h"
#include "input.h"
#include "config.h"
#include "timer.h"
#include "formats.h"

#define max(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a > _b ? _a : _b; })

#define min(a,b) \
({ __typeof__ (a) _a = (a); \
   __typeof__ (b) _b = (b); \
 _a < _b ? _a : _b; })


 void usage(int argc, char** argv);

// MIN_ITER, MAX_ITER, TIME_LIMIT, 
double benchmark_coo_spmv(coo_matrix * coo, float* x, float* y);

