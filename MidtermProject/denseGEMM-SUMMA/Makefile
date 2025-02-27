ifeq ($(OS),Windows_NT)
    MPI_PATH = C:/Program Files/Microsoft MPI/Bin
    #CC = "$(MPI_PATH)/mpicc.exe"
	CC = "$(MPI_PATH)/mpiexec.exe"
else
    CC = mpicc
endif
CFLAGS = -Wall -O3
INCLUDES = -I./include
LDFLAGS = -lm

SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj
BIN_DIR = .

SOURCES = $(wildcard $(SRC_DIR)/*.c)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
TARGET = summa

all: directories $(TARGET) run-tests

directories:
ifeq ($(OS),Windows_NT)
	if not exist "$(OBJ_DIR)" mkdir "$(OBJ_DIR)"
else
	mkdir -p $(OBJ_DIR)
endif

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(INC_DIR)/*.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Tests compilation
tests.exe: tests.c $(OBJECTS)
	gcc $(CFLAGS) $(INCLUDES) $^ -o $@ $(LDFLAGS)

run:
	mpirun -np 4 ./summa -m 16 -n 16 -k 16 -b 16 -s b -v -p

run-tests: tests.exe
	./tests.exe

clean: 
ifeq ($(OS),Windows_NT)
	del /f *.o 2>nul || true
	del /f summa.exe 2>nul || true
	del /f tests.exe 2>nul || true
else
	find ./ -name "*.o" -delete
	rm -rf $(OBJ_DIR) $(TARGET)
endif

.PHONY: all clean run directories run-tests