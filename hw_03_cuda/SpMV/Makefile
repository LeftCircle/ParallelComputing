NVCC = nvcc
NVFLAGS = -O3 -std=c++11 -I./include/ -diag-suppress 177,2464

all: spmv-cuda spmv-mpi-cuda

spmv-cuda: spmv-cuda.cu mmio.c
	$(NVCC) $(NVFLAGS) -x c mmio.c -x cu spmv-cuda.cu -o $@

spmv-mpi-cuda: spmv-mpi-cuda.cu mmio.c
	$(NVCC) $(NVFLAGS) -x c mmio.c -x cu spmv-mpi-cuda.cu -ccbin mpicxx -Xcompiler "-DMPICH_SKIP_MPICXX" -o $@ -lmpi

clean:
	rm -f spmv-cuda spmv-mpi-cuda

hybrid:
	mpirun -np 1 ./spmv-mpi-cuda test_matrices/D6-6.mtx

cuda:
	./spmv-cuda test_matrices/D6-6.mtx


# NVCC = nvcc
# NVFLAGS = -O3 -std=c++11 -I./include/ -diag-suppress=177,2464
# CC = gcc
# CFLAGS = -O3 -fPIC -I./include/ -Wno-null-character

# # Object files
# CUDA_OBJS = spmv-cuda.o mmio.o
# DEPS = include/mmio.h

# all: spmv-cuda

# # Main target - specify source files explicitly
# # spmv-cuda: spmv-cuda.o mmio.o
# # 	$(NVCC) $(NVFLAGS) -x cu spmv-cuda.o mmio.o -o $@

# # # Compile CUDA source
# # spmv-cuda.o: spmv-cuda.cu $(DEPS)
# # 	$(NVCC) $(NVFLAGS) -c $< -o $@

# # # Compile C source - compile as C, not C++
# # mmio.o: mmio.c $(DEPS)
# # 	$(CC) $(CFLAGS) -x c -c $< -o $@

# # Main target
# spmv-cuda: $(CUDA_OBJS)
# 	$(NVCC) $(NVFLAGS) $^ -o $@

# # Compile CUDA source
# spmv-cuda.o: spmv-cuda.cu $(DEPS)
# 	$(NVCC) $(NVFLAGS) -c $< -o $@

# # Compile C source
# mmio.o: mmio.c $(DEPS)
# 	$(CC) $(CFLAGS) -c $< -o $@

# clean:
# 	rm -f spmv-cuda spmv-mpi-cuda mmio.o

# cuda:
# 	./spmv-cuda test_matrices/D6-6.mtx

# mpi:
# 	mpirun -np 1 ./spmv-mpi-cuda test_matrices/D6-6.mtx