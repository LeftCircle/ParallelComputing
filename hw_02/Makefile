CC=mpicc
#CC=mpicc
#FLAG=-g -Wall
FLAG=-O3 -std=c99 -I./include/ -Wno-unused-result -Wno-write-strings -fopenmp
LDFLAG=-O3 -fopenmp

OBJS=spmv-mpi.o mmio.o 

#all: spmv-mpi tests.exe spmv-omp run-tests
all: spmv-mpi spmv-omp tests.exe run-tests

.c.o:
	${CC} -o $@ -c ${FLAG} $<

spmv-mpi: ${OBJS}
	mpicc  ${LDFLAG} -o $@ $^

# Tests compilation
tests.exe: tests.o 
	${CC} ${LDFLAG} -o $@ $^

# spmv-omp compilation
spmv-omp: spmv-omp.o mmio.o
	gcc ${LDFLAG} -o $@ $^

.PHONY:run-tests
run-tests: tests.exe
	./tests.exe

.PHONY:clean
clean: 
ifeq ($(OS),Windows_NT)
	del /f *.o 2>nul || true
	del /f spmv-mpi.exe 2>nul || true
	del /f tests.exe 2>nul || true
else
	find ./ -name "*.o" -delete
	rm -f spmv-mpi tests.exe
endif

