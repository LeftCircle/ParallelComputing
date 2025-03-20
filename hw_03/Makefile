CC = g++

CCFLAGS = -std=c++2a -O3 -ffast-math -msse4.2 -fopenmp -lm 

INCLUDES = -I./include

SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj

SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

TARGET = stencil

all: directories stencil

directories:
ifeq ($(OS),Windows_NT)
	if not exist "$(OBJ_DIR)" mkdir "$(OBJ_DIR)"
else
	mkdir -p $(OBJ_DIR)
endif

# stencil: stencil.cpp
# 	$(CC) $(CCFLAGS) -o $@ $<

stencil: $(OBJECTS)
	$(CC) $(CCFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(INC_DIR)/*.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

run:
	./stencil

test:
	./stencil -t

clean: 
ifeq ($(OS),Windows_NT)
	del /f *.o 2>nul || true
	del /f summa.exe 2>nul || true
	del /f tests.exe 2>nul || true
else
	find ./ -name "*.o" -delete
	rm -rf $(OBJ_DIR) $(TARGET)
endif



# clean:
# 	rm -rf stencil