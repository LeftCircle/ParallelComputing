#!/bin/bash

k_vals=(16 256 512 1024)
grid_size=2048

# Make sure the program is compiled
make clean && make

# Loop through different configurations
for k in "${k_vals[@]}"
do
    echo "Running stencil with grid size ${grid_size}x${grid_size}, k: ${k}"
    ./stencil $grid_size $k
    echo "----------------------------------------"
done


