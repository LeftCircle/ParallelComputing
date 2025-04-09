CC = g++

CCFLAGS = -std=c++2a -O3 -ffast-math -msse4.2 -fopenmp -lm -g
#Add OpenGL and freeglut
LDFLAGS = -lGL -lGLU -lglut -lm
INCLUDES = -I./include -I/usr/include/eigen3

SRC_DIR = source
INC_DIR = include
OBJ_DIR = obj

SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

TARGET = boids

all: directories $(TARGET)

directories:
ifeq ($(OS),Windows_NT)
	if not exist "$(OBJ_DIR)" mkdir "$(OBJ_DIR)"
else
	mkdir -p $(OBJ_DIR)
endif

$(TARGET): $(OBJECTS)
	$(CC) $(CCFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(INC_DIR)/*.h
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

run:
	./boids

test:
	./boids -t

clean: 
ifeq ($(OS),Windows_NT)
	del /f *.o 2>nul || true
else
	find ./ -name "*.o" -delete
	rm -rf $(OBJ_DIR) $(TARGET)
endif

