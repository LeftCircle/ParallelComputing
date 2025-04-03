from pyspark.sql import SparkSession
from pyspark.sql.SparkSession import Broadcast
from pyspark.rdd import RDD
import numpy as np
from scipy.io import mmread
from scipy import sparse
from typing import List, Tuple, Any
import time
import path_utils
import argparse

def read_coo_matrix(filename: str) -> sparse.coo_matrix:
	matrix = mmread(filename)
	return matrix.tocoo()

def sequential_spmv(coo_matrix: sparse.coo_matrix,
					x_vector: np.ndarray) -> np.ndarray:
	"""Sequential Sparse Matrix-Vector Multiplication (SpMV)."""
	time_start = time.time()
	result = np.zeros(coo_matrix.shape[0])
	for i, j, v in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
		result[i] += v * x_vector[j]
	time_end = time.time()
	print(f"Sequential SpMV time: {time_end - time_start:.4f} seconds")
	return result

def multiply_row(row_data: Tuple[int, List[Tuple[int, float]]],
                 x_broadcast: Broadcast) -> Tuple[int, float]:
	"""Multiply row elements with corresponding x values"""
	row_id, col_val_pairs = row_data
	result: float = 0.0
	for col, val in col_val_pairs:
		result += val * x_broadcast.value[col]
	return (row_id, result)

def spmv_coo_spark_row(spark: SparkSession,
                   coo_matrix: sparse.coo_matrix,
                   x_vector: np.ndarray) -> np.ndarray:
	"""Sparse matrix-vector multiplication using COO format in Spark."""
	
	# Convert COO matrix to RDD of (row, (col, val)) pairs
	matrix_rdd: RDD = spark.sparkContext.parallelize(
		zip(coo_matrix.row, coo_matrix.col, coo_matrix.data)
	)
	matrix_rdd = matrix_rdd.map(lambda x: (x[0], (x[1], x[2])))
	
	# Broadcast the vector x
	x_broadcast: Broadcast = spark.sparkContext.broadcast(x_vector)
	
	time_start = time.time()
	# Perform the multiplication on each row
	result_rdd: List[Tuple[int, float]] = matrix_rdd.groupByKey()\
						   				 .map(lambda x: multiply_row(x, x_broadcast))\
						   				 .collect()
	time_end = time.time()
	print(f"Spark SpMV time: {time_end - time_start:.4f} seconds")
	
	# cleanup
	x_broadcast.unpersist()

	#unpack_time = time.time()
	# Collect results into a dense array
	result: np.ndarray = np.zeros(coo_matrix.shape[0])
	for row_id, value in result_rdd:
		result[row_id] = value
	
	#unpack_time = time.time() - unpack_time
	#print(f"Unpacking time: {unpack_time * 1000:.4f} ms")
	return result


def spmv_coo_spark(spark: SparkSession,
                   coo_matrix: sparse.coo_matrix,
                   x_vector: np.ndarray) -> np.ndarray:
	"""Sparse matrix-vector multiplication using COO format in Spark."""
	
	# Parallelize COO matrix to RDD of (row, col, val)
	matrix_rdd: RDD = spark.sparkContext.parallelize(
		zip(coo_matrix.row, coo_matrix.col, coo_matrix.data)
	)
	
	# Broadcast the vector x
	x_broadcast: Broadcast = spark.sparkContext.broadcast(x_vector)
	
	time_start = time.time()
	partial_rdd = matrix_rdd.map(lambda x: (x[0], x[2] * x_broadcast.value[x[1]]))
	result_rdd: List[Tuple[int, float]] = partial_rdd.reduceByKey(lambda a, b: a + b)\
						   				 .collect()

	time_end = time.time()
	print(f"Spark SpMV time: {time_end - time_start:.4f} seconds")
	
	# cleanup
	x_broadcast.unpersist()

	#unpack_time = time.time()
	# Collect results into a dense array
	result: np.ndarray = np.zeros(coo_matrix.shape[0])
	for row_id, value in result_rdd:
		result[row_id] = value
	
	#unpack_time = time.time() - unpack_time
	#print(f"Unpacking time: {unpack_time * 1000:.4f} ms")
	return result

def benchmark_spmv_spark(spark: SparkSession,
                		 coo_matrix: sparse.coo_matrix,
                		 x_vector: np.ndarray,
						 num_iterations=100) -> float:
	"""Benchmark SpMV using Spark with COO format."""

	# Convert COO matrix to RDD of (row, (col, val)) pairs
	matrix_rdd: RDD = spark.sparkContext.parallelize(
		zip(coo_matrix.row, coo_matrix.col, coo_matrix.data)
	)
	matrix_rdd = matrix_rdd.map(lambda x: (x[0], (x[1], x[2])))
	
	# Broadcast the vector x
	x_broadcast = spark.sparkContext.broadcast(x_vector)
	
	# Warmup
	start_time = time.time()
	result_rdd: List[Tuple[int, float]] = matrix_rdd.groupByKey()\
						   				 .map(lambda x: multiply_row(x, x_broadcast))\
						   				 .collect()
	warmup_time = time.time() - start_time
	print(f"\tWarmup time: {warmup_time * 1000:.4f} ms")

	# Benchmark
	times = []
	for _ in range(num_iterations):
		start_time = time.time()
		result_rdd: List[Tuple[int, float]] = matrix_rdd.groupByKey()\
						   				 .map(lambda x: multiply_row(x, x_broadcast))\
						   				 .collect()
		times.append(time.time() - start_time)
	
	avg_time = np.mean(times) * 1000 
	gflops = (2.0 * coo_matrix.nnz / (avg_time / 1000)) / 1e9
	
	# Cleanup
	x_broadcast.unpersist()

	print(f"\tPerforming {num_iterations} iterations")
	print(f"\tbenchmarking Spark-SpMV: {avg_time:8.4f} ms ({gflops:5.2f} GFLOP/s)")
	
	return avg_time

def test_spmv_spark():
	"""Test the SpMV implementation using Spark."""
	
	spark = SparkSession.builder\
			.appName("SpMV")\
			.getOrCreate()
	
	# Read matrix
	#file = path_utils.get_full_path_from_relative_path("test_matrices/D6-6.mtx")
	file = path_utils.get_full_path_from_relative_path("test_matrices/bfly.mtx")
	matrix = read_coo_matrix(file)
	
	# Create random vector x
	x = np.random.random(matrix.shape[1])
	
	spark_result: np.ndarray = spmv_coo_spark(spark, matrix, x)
	sequential_result: np.ndarray = sequential_spmv(matrix, x)
	assert np.allclose(spark_result, sequential_result), "Results do not match!"
	print("Results match!")
	
	spark.stop()

def main():
	parser = argparse.ArgumentParser(description='Sparse Matrix-Vector Multiplication using Spark')
	parser.add_argument('matrix_path', type=str, help='Path to the matrix market file (.mtx)')
	args = parser.parse_args()
	# Read matrix
	#matrix_dir: str = path_utils.get_full_path_from_relative_path("test_matrices")
	#matrices: List[str] = path_utils.get_all_files_of_suffix(matrix_dir, ".mtx")
	#for matrix_file in matrices:
	matrix_file: str = path_utils.get_full_path_from_relative_path(args.matrix_path)
	matrix_file_name = path_utils.get_file_name_without_extension(matrix_file)
	print(f"Matrix: {matrix_file_name}")
		# Read matrix
		# Initialize Spark
	spark = SparkSession.builder\
			.appName("SpMV")\
			.getOrCreate()
	matrix = read_coo_matrix(matrix_file)
	
	# Create random vector x
	x = np.random.random(matrix.shape[1])
	
	# Run benchmark
	avg_time: float = benchmark_spmv_spark(spark, matrix, x, 3)
	print(f"Average time: {avg_time:.4f} ms")

	spark.stop()

if __name__ == "__main__":
	#main()
	test_spmv_spark()