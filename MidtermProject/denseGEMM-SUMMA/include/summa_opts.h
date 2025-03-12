#ifndef __SUMMA_OPTS_H__
#define __SUMMA_OPTS_H__

typedef struct {
    int m;              // rows of A
    int n;              // cols of B
    int k;              // cols of A/rows of B
    int block_size;     // block size for tiling
    char stationary;    // 'a' or 'b' for algorithm variant
    int verbose;        // print additional info
    int metrics;        // print performance metrics
} SummaOpts;

typedef struct {
	int rows;
	int cols;
	float* matrix;
} Matrix;

void print_usage(const char* prog_name);
SummaOpts parse_args(int argc, char *argv[]);

#endif