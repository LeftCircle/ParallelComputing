#!/bin/bash

# Loop through all files in test_matrices directory
for matrix in test_matrices/*.mtx; do
    if [ -f "$matrix" ]; then
        echo "Running spmv-spark with matrix: $matrix"
        python3 spmv-spark.py "$matrix"
        echo "----------------------------------------"
    fi
done