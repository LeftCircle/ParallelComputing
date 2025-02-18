CC=mpicc
#CC=mpicc
#FLAG=-g -Wall
FLAG=-O3 -std=c99 -I./include/ -Wno-unused-result -Wno-write-strings -fopenmp
LDFLAG=-O3 -fopenmp

OBJS= mmio.o spmv.o

#all: spmv-mpi tests.exe spmv-omp run-tests
all: spmv-mpi spmv-omp tests.exe run-tests

.c.o:
	${CC} -o $@ -c ${FLAG} $<

spmv.o: spmv.c spmv.h
	${CC} -o $@ -c ${FLAG} $<

mmio.o: mmio.c
	${CC} -o $@ -c ${FLAG} $<

spmv-mpi.o: spmv-mpi.c spmv.h
	mpicc -o $@ -c ${FLAG} $<

spmv-mpi: spmv-mpi.o ${OBJS}
	mpicc  ${LDFLAG} -o $@ $^

# Tests compilation
tests.exe: tests.o 
	${CC} ${LDFLAG} -o $@ $^

spmv-omp.o: spmv-omp.c spmv.h
	${CC} -o $@ -c ${FLAG} $<

# spmv-omp compilation
spmv-omp: spmv-omp.o ${OBJS}
	${CC} ${LDFLAG} -o $@ $^

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
	rm -f spmv-mpi tests.exe spmv-omp
endif

