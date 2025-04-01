## 2D Stencil SIMD and SIMD+OpenMP Implementation

### Overview

The stencil operation processes a 2D grid where each point's new value is calculated based on its neighboring points within a radius K. The implementation uses SIMD instructions to speed up the summation of neighbors.

### Visual Representation
For a grid point (x,y) with K=2:

```
    □ □ ▲ □ □
    □ □ ▲ □ □
    ◄ ◄ ● ► ►
    □ □ ▼ □ □
    □ □ ▼ □ □

● = Current point (x,y)
▲ = Top neighbors
▼ = Bottom neighbors
◄ = Left neighbors
► = Right neighbors
□ = Points outside stencil
```

### SIMD Processing
1. For each point (x,y), the algorithm:
  - Processes neighbors in 4 directions using SIMD instructions
  - Uses 128-bit SSE registers (__m128) to process 4 floats at once
2. SIMD accumulation for each direction:

```
[n1][n2][n3][n4] → __m128 register
     +
[n5][n6][n7][n8] → __m128 register
     =
[sum of 4 values at a time]
```

### Memory Layout
```
Row-major memory layout:
[0,0][0,1][0,2]...[0,N]  ← Process horizontally with SIMD
[1,0][1,1][1,2]...[1,N]
[2,0][2,1][2,2]...[2,N]
  ...
[N,0][N,1][N,2]...[N,N]

Trans array for vertical access:
[0,0][1,0][2,0]...[N,0]  ← Process vertically with SIMD
[0,1][1,1][2,1]...[N,1]
[0,2][1,2][2,2]...[N,2]
  ...
[0,N][1,N][2,N]...[N,N]
```

### Key Features
- Aligned memory allocation (64-byte alignment)
- Unaligned SIMD loads for flexibility
- Boundary handling with partial loads
- Separate transposed array for efficient vertical summation
- Vectorized summation using SSE instructions

### Performance Optimizations
- Uses __restrict__ pointers to enable better compiler optimizations
- Processes 4 elements at once using SSE
- Handles edge cases efficiently with partial loads
- Minimizes memory movement with in-place updates

### Discussion
For the following problem, the ideal block size seemed to have little effect on performance, although the blocked solution was always faster than the non-blocked versions. The blocked solutions resulted in better cache utilization since BxB blocks were processed at a time. With a large enough block size, this would result in more cache misses as an entire block may no longer fit in a cache line. 

### Performance optimizations with OpenMP:

Sequential Execution (Without OpenMP)
```
Thread 0:
┌────┬────┬────┬────┐
│BBBB│BBBB│BBBB│BBBB│→ Sequential processing
│BBBB│BBBB│BBBB│BBBB│→ of blocks
│BBBB│BBBB│BBBB│BBBB│→
│BBBB│BBBB│BBBB│BBBB│→
└────┴────┴────┴────┘
```

Parallel Execution with OpenMP
```
Thread 0:          Thread 1:
┌────┬────┐       ┌────┬────┐
│BBBB│BBBB│       │BBBB│BBBB│
│BBBB│BBBB│       │BBBB│BBBB│
└────┴────┘       └────┴────┘

Thread 2:          Thread 3:
┌────┬────┐       ┌────┬────┐
│BBBB│BBBB│       │BBBB│BBBB│
│BBBB│BBBB│       │BBBB│BBBB│
└────┴────┘       └────┴────┘
```
The combination of blocking (B), SIMD operations, and OpenMP parallelization provides three levels of optimization:

- Cache efficiency (blocking)
- Vectorization (SIMD)
- Multi-core utilization (OpenMP)


### Results
can be found in results.txt and the attached images. 