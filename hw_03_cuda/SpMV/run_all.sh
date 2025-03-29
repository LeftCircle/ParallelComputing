#!/bin/bash

# Loop through all files in test_matrices directory
for matrix in test_matrices/*.mtx; do
    if [ -f "$matrix" ]; then
        echo "Running spmv-cuda with matrix: $matrix"
        ./spmv-cuda "$matrix"
        echo "----------------------------------------"
    fi
done