CC=gcc
#FLAG=-g -Wall
FLAG=-O3 -std=c99 -I./include/ -Wno-unused-result -Wno-write-strings
LDFLAG=-O3

OBJS=spmv-mpi.o mmio.o 

all: spmv-mpi tests.exe run-tests

.c.o:
	${CC} -o $@ -c ${FLAG} $<

spmv-mpi: ${OBJS}
	${CC}  ${LDFLAG} -o $@ $^

# Tests compilation
tests.exe: tests.o 
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
	rm -f spmv-mpi tests.exe
endif

