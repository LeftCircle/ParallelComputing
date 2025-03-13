# MPI Hello World

This project is a simple MPI (Message Passing Interface) program that demonstrates how to set up and run an MPI application. Each process will print a "Hello, World!" message.

## Files

- `src/main.c`: Contains the main function for the MPI program.
- `Makefile`: Contains the build instructions for the project.

## Building the Project

To build the project, navigate to the project directory and run the following command:

```
make
```

This will compile the `main.c` file into an executable.

## Running the Program

To run the program using MPI, use the following command:

```
mpirun -np <number_of_processes> ./main
```

Replace `<number_of_processes>` with the desired number of processes you want to run.

## Example

To run the program with 4 processes, use:

```
mpirun -np 4 ./main
```

## Requirements

Make sure you have an MPI implementation installed (e.g., OpenMPI or MPICH) to compile and run this program.