# [HW2] MPI and Shared-Memory Programming (100 points)

This repo contains my solutions to HW2, MPI and OpenMP programming. There are three main scripts:
 1. spmv-mpi.c
 2. spmv-omp.c
 3. spmv-hybrid.c

Each of these solutions benchmarks the performance of coo-spmv calculations using different parallel computing approaches. The results are compared to the sequential version of the calculation.
The following combinations were used:
1. MPI (with 8 nodes and one core per node)
2. OpenMP (With 8 cores utilized)
3. Hybrid_N4_n8 (Four nodes with two cores each. Each OpenMP instruction used 2 threads, for a total of 8 parallel regions)
4. Hybrid_N8_n8 (8 nodes with one core each. Each OpenMP instruction used 2 threads, for a total of 16 parallel regions)

Q: I do have some existing questions regarding the number of parallel regions utilized for the hybrid solutions. I'm not entirely sure if OpenMP will utilize more cores to spin up more parallel threads when the total number of parallel regions is greater than the number of nodes and cores that were grabbed through MPI. The graph seems to show that both Hybrid solutions may have only used 8 parallel regions, where Hybrid_n8_n4 was creating more OpenMP threads than were available. 

## Results
The results can be seen from the following images. One thing to note is that for the MPI calculations, when the transfer time to split the data from node 0 to other nodes, the latency accounted for the majority of the computation time. This was not observed when the reduction of the results from the other nodes to node 0 was included in the benchmark times. Excluding the startup times from the calculation makes the MPI
Another notable piece is that the sequential algorithm outperformed almost all methods for smaller matrix sizes. The size of the matrices is not included in the graphic, but more information on them can be found from the link above. 

![With MPI Data Init](https://github.com/LeftCircle/ParallelComputing/blob/main/hw_02/graphs/TimesIncludingDataTransfer.png)

![Without MPI Data Init](https://github.com/LeftCircle/ParallelComputing/blob/main/hw_02/graphs/NoTransferTimes.png)


## Background
Sparse Matrix Vector multiplication (short for SpMV) is an important kernel in both traditional high performance computing and emerging data-intensive applications and deep neural networks after using pruning techniques. Since the data is sparse, that could come from noisy data, missing data, etc., efficient data representation (a.k.a. storage format) without storing zero elements and algorithm is needed to improve its performance. Due to the sparsity, SpMV algorithm generally has irregular memory access, which increases difficulty in performance improvement.
    
 SpMV is to compute vector `y` by multiplying sparse matrix `A` and dense vector `x`. Since the output vector `y` is generally dense, we also use dense format to save it. Thus, `x` and `y` are in dense format, stored as two arrays; while sparse matrix `A` is stored in a sparse representation. In this assignment, we choose a very traditional and simple Coordinate (COO) format to represent the sparse matrix `A`. COO format has three arrays to represent a sparse matrix: `rows`, `cols`, and `vals`. The COO format explicitly stores the row indices. The `rows`, `cols` and `vals` arrays store the row indices, the column indices, and the values of all nonzero elements respectively.

Testing Dataset: SuiteSparse: [http://sparse.tamu.edu](http://sparse.tamu.edu/) . Download matrices:D6-6, dictionary28, Ga3As3H12, bfly, pkustk14, roadNet-CA.

## spm-mpi.c

### MPI Data Distribution Pattern

```ascii
                                Node 0 (Master)
                    ┌───────────────────────────────────┐
                    │ Complete COO Matrix Data:         │
                    │ ├── rows[]                        │
                    │ ├── cols[]                        │
                    │ ├── vals[]                        │
                    │ ├── x[] (complete vector)         │
                    │ ├── y[] (holds local result)      │
                    | └── y_parallel[] (final result)   |
                    └─────────────────┬─────────────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     │   Data Distribution Phase     │
                     └───────────────┬───────────────┘
                                    ↓
         ┌──────────────────────────┼──────────────────────────┐
         ↓                          ↓                          ↓
    Node 1                     Node 2                     Node 3
┌─────────────┐           ┌───────────────┐           ┌─────────────┐
│ Partial:    │           │ Partial:      │           │ Partial:    │
│ rows[0:n/3] │           │ rows[n/3:2n/3]│           │ rows[2n/3:n]│
│ cols[0:n/3] │           │ cols[n/3:2n/3]│           │ cols[2n/3:n]│
│ vals[0:n/3] │           │ vals[n/3:2n/3]│           │ vals[2n/3:n]│
│             │           │               │           │             │
│ Complete:   │           │ Complete:     │           │ Complete:   │
│ x[] vector  │           │ x[] vector    │           │ x[] vector  │
└─────────────┘           └───────────────┘           └─────────────┘

Data Distribution Method:
1. MPI_Scatter:  Distributes workload sizes. Other nodes allocate space.
2. MPI_Scatterv: Distributes rows[], cols[], vals[] arrays
3. MPI_Bcast:    Broadcasts complete x[] vector and space required for y vector
4. MPI_Reduce:   Combines partial results back to Node 0
```

### Distribution Process Details

1. **Initial Setup (Node 0)**
   - Reads complete matrix
   - Calculates workload distribution
   - Allocates arrays

2. **Workload Distribution**
   - Uses `split_workload()` to determine how many elements each node gets
   - Creates displacement array for scattering data

3. **Data Transfer Operations**
   - `MPI_Scatter`: Sends workload sizes
   - `MPI_Scatterv`: Distributes matrix data (rows, cols, vals)
   - `MPI_Bcast`: Shares space required for x and y vector, then Shares complete x vector 

4. **Computation**
   - Each node performs SpMV on its portion
   - Results combined using `MPI_Reduce`

5. **Final Result**
   - Node 0 receives combined y vector
   - Other nodes free their temporary memory


## spi-omp.c

For benchmarking, I chose to use `#pragma omp parallel for reduction(+:y[:coo->num_rows]) num_threads(N_THREADS)`.
There is a slightly slower manual implementation of the private y vectors per thread and the reduction process.

### OpenMP Parallel Reduction Pattern for SpMV

```ascii
Input Matrix (COO Format)       Input Vector x
rows: [0,2,1,2,1]                [x0]
cols: [0,0,1,1,2]                [x1]
vals: [1,2,3,4,5]                [x2]

                    Thread Distribution
                           ↓
┌──────────────────────────────────────────────────┐
│              #pragma omp parallel for            │
│         reduction(+:y[:coo->num_rows])           │
└─────────────────┬───────────────┬────────────────┘
                  │               │
        ┌─────────┴─────┬─────────┴────────┐
        ↓               ↓                   ↓
    Thread 0        Thread 1           Thread 2
   Iterations:     Iterations:        Iterations:
   [0,1]           [2,3]             [4]
    
    y[0]+=1*x[0]   y[1]+=3*x[1]      y[1]+=5*x[2]
    y[2]+=2*x[0]   y[2]+=4*x[1]

                    Final y Vector
                   (After Reduction)
                        [y0]
                        [y1]
                        [y2]

Key Features:
- Each thread processes its assigned iterations
- Reduction clause automatically handles race conditions
- No manual result combining needed
- Threads synchronize at the end of parallel region
```

### How the Reduction Works:

1. **Initial Setup**
   - OpenMP creates private copies of y array for each thread
   - Each thread gets a subset of iterations

2. **Parallel Execution**
   - Threads work independently on their portions
   - No locks needed due to reduction clause
   - Private y arrays updated separately

3. **Automatic Reduction**
   - OpenMP combines private y arrays at end
   - Uses efficient tree-based reduction
   - Results combined into shared y array

4. **Advantages**
   - No manual thread management needed
   - Efficient handling of race conditions
   - Better performance than critical sections
   - Simpler code than manual thread version


## spmv-hybrid

### Hybrid MPI + OpenMP SpMV Implementation

```ascii
                        MPI Node 0 (Master)
                 ┌─────────────────────────────┐
                 │ Complete Matrix Data        │
                 │ rows[], cols[], vals[]      │
                 └───────────────┬─────────────┘
                                │
                 MPI Distribute │
          ┌────────────────────┼────────────────────┐
          ↓                    ↓                    ↓
    MPI Node 1           MPI Node 2           MPI Node 3
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Partial Matrix: │ │ Partial Matrix: │ │ Partial Matrix: │
│ ~33% of data    │ │ ~33% of data    │ │ ~33% of data    │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                    │                    │
   OpenMP Teams        OpenMP Teams         OpenMP Teams
         │                    │                    │
    ┌────┴────┐         ┌────┴────┐         ┌────┴────┐
    ↓         ↓         ↓         ↓         ↓         ↓
Thread 1   Thread 2  Thread 1  Thread 2  Thread 1   Thread 2

Data Flow:
1. MPI Level (Coarse-Grain Parallelism)
   └─► Distributes matrix chunks between nodes
   └─► Broadcasts complete x vector
   └─► Reduces final y results

2. OpenMP Level (Fine-Grain Parallelism)
   └─► Each MPI node splits its chunk between threads
   └─► Uses reduction clause for thread synchronization
   └─► Threads process their portion independently

Example for 3 MPI nodes × 2 OpenMP threads:
┌────────────────────────────────────────────┐
│ Total Parallel Units = 6                   │
│ Parallelism Levels:                        │
│ └─► Level 1: MPI (Distributed Memory)      │
│     - Handles data distribution            │
│     - Manages inter-node communication     │
│                                            │
│ └─► Level 2: OpenMP (Shared Memory)        │
│     - Processes local data chunks          │
│     - Uses efficient reduction             │
└────────────────────────────────────────────┘
```

### Key Implementation Features

1. **MPI Layer** 
   - Handles distributed memory operations
   - Splits matrix between nodes using `_split_vals_between_nodes()`
   - Broadcasts x vector using `_broadcast_data_for_x_and_y()`
   - Reduces final results with `MPI_Reduce()`

2. **OpenMP Layer**
   - Manages shared memory parallelism within each node
   - Uses `#pragma omp parallel for reduction`
   - Each thread processes subset of local matrix elements
   - Automatic thread synchronization via reduction clause

3. **Performance Benefits**
   - Exploits both inter-node and intra-node parallelism
   - Reduces communication overhead vs pure MPI
   - Better memory utilization vs pure OpenMP
   - Scalable across clusters with multicore nodes