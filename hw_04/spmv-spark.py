from pyspark.sql import SparkSession
#from pyspark.sql.SparkSession import Broadcast
from pyspark.rdd import RDD
import numpy as np
from scipy.io import mmread
from scipy import sparse
from typing import List, Tuple, Any
import time
import path_utils
import argparse
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
import ctypes
import os


def spmv_c(coo_matrix: sparse.coo_matrix, x_vector: np.ndarray) -> np.ndarray:
	"""
	Call C implementation of SpMV

	Parameters:
		coo_matrix: Sparse matrix in COO format
		x_vector: Input vector
		
	Returns:
		Result vector
	"""
	# Load the shared library
	lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libspmv.so")
	spmv_lib = ctypes.CDLL(lib_path)

	# Define argument types
	spmv_lib.coo_spmv.argtypes = [
		ctypes.c_int,        # n_rows
		ctypes.c_int,        # n_cols
		ctypes.c_int,        # nnz
		ctypes.POINTER(ctypes.c_int),    # rows
		ctypes.POINTER(ctypes.c_int),    # cols
		ctypes.POINTER(ctypes.c_float),  # vals
		ctypes.POINTER(ctypes.c_float),  # x
		ctypes.POINTER(ctypes.c_float)   # y
	]

	# Prepare data
	n_rows, n_cols = coo_matrix.shape
	nnz = coo_matrix.nnz

	# Create C-compatible arrays
	rows_array = coo_matrix.row.astype(np.int32)
	cols_array = coo_matrix.col.astype(np.int32)
	vals_array = coo_matrix.data.astype(np.float32)
	x_array = x_vector.astype(np.float32)
	y_array = np.zeros(n_rows, dtype=np.float32)

	# Create C pointers
	rows_ptr = rows_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
	cols_ptr = cols_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
	vals_ptr = vals_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
	x_ptr = x_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
	y_ptr = y_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

	# Call C function
	time_start = time.time()
	spmv_lib.coo_spmv(n_rows, n_cols, nnz, rows_ptr, cols_ptr, vals_ptr, x_ptr, y_ptr)
	time_end = time.time()
	print(f"C SpMV time: {time_end - time_start:.4f} seconds")

	return y_array

def read_coo_matrix(filename: str) -> sparse.coo_matrix:
	matrix = mmread(filename)
	return matrix.tocoo()

def sequential_spmv(coo_matrix: sparse.coo_matrix,
					x_vector: np.ndarray) -> np.ndarray:
	time_start = time.time()
	result = np.zeros(coo_matrix.shape[0])
	for i, j, v in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
		result[i] += v * x_vector[j]
	time_end = time.time()
	print(f"Sequential SpMV time: {time_end - time_start:.4f} seconds")
	return result

def multiply_row(row_data: Tuple[int, List[Tuple[int, float]]],
                 x_broadcast) -> Tuple[int, float]:
	row_id, col_val_pairs = row_data
	result: float = 0.0
	for col, val in col_val_pairs:
		result += val * x_broadcast.value[col]
	return (row_id, result)

def spmv_coo_spark_row(spark: SparkSession,
					   coo_matrix: sparse.coo_matrix,
					   x_vector: np.ndarray) -> np.ndarray:	
	# Convert COO matrix to RDD of (row, (col, val)) pairs
	matrix_rdd: RDD = spark.sparkContext.parallelize(
		zip(coo_matrix.row, coo_matrix.col, coo_matrix.data)
	)
	matrix_rdd = matrix_rdd.map(lambda x: (x[0], (x[1], x[2])))
	
	# Broadcast the vector x
	x_broadcast = spark.sparkContext.broadcast(x_vector)
	
	time_start = time.time()
	# Perform the multiplication on each row
	result_rdd: List[Tuple[int, float]] = matrix_rdd.groupByKey()\
										  .map(lambda x: multiply_row(x, x_broadcast))\
										  .collect()
	time_end = time.time()
	print(f"Spark SpMV time row: {time_end - time_start:.4f} seconds")
	
	# cleanup
	x_broadcast.unpersist()

	# Collect results into a dense array
	result: np.ndarray = np.zeros(coo_matrix.shape[0])
	for row_id, value in result_rdd:
		result[row_id] = value
	
	return result


def spmv_coo_spark(spark: SparkSession,
				   coo_matrix: sparse.coo_matrix,
				   x_vector: np.ndarray) -> np.ndarray:	
	# Parallelize COO matrix to RDD of (row, col, val)
	matrix_rdd: RDD = spark.sparkContext.parallelize(
		zip(coo_matrix.row, coo_matrix.col, coo_matrix.data)
	)
	
	# Broadcast the vector x
	x_broadcast = spark.sparkContext.broadcast(x_vector)
	
	time_start = time.time()
	partial_rdd = matrix_rdd.map(lambda x: (x[0], x[2] * x_broadcast.value[x[1]]))
	result_rdd: List[Tuple[int, float]] = partial_rdd.reduceByKey(lambda a, b: a + b)\
										  .collect()

	time_end = time.time()
	print(f"Spark SpMV time: {time_end - time_start:.4f} seconds")
	
	# cleanup
	x_broadcast.unpersist()

	# Collect results into a dense array
	result: np.ndarray = np.zeros(coo_matrix.shape[0])
	for row_id, value in result_rdd:
		result[row_id] = value
	
	return result

def spmv_coo_spark_dataframe(spark: SparkSession,
							 coo_matrix: sparse.coo_matrix,
							 x_vector: np.ndarray) -> np.ndarray:
	time_start = time.time()

	# Define explicit schema
	schema = StructType([
		StructField("row", IntegerType(), False),
		StructField("col", IntegerType(), False),
		StructField("val", FloatType(), False)
	])

	# Convert to Python types from numpy
	row_data = [(int(r), int(c), float(v)) for r, c, v in 
				zip(coo_matrix.row, coo_matrix.col, coo_matrix.data)]

    # Create DataFrame with explicit schema
	matrix_df = spark.createDataFrame(row_data, schema=schema)
	
	# Broadcast the vector x
	x_broadcast = spark.sparkContext.broadcast(x_vector)

	# Use a UDF to access broadcast variable
	@udf(FloatType())
	def vector_lookup(col_idx):
		return float(x_broadcast.value[int(col_idx)])
	
	partial_df = matrix_df.withColumn("x_val", col("val") * vector_lookup(col("col")))
	result_df = partial_df.groupBy("row").agg({"x_val": "sum"}).select("row", "sum(x_val)")
	
	time_end = time.time()
	print(f"\tdataframe time : {(time_end - time_start):.4f} s")
	
	# cleanup
	x_broadcast.unpersist()

	# Collect results into a dense array
	result: np.ndarray = np.zeros(coo_matrix.shape[0])
	for row in result_df.collect():
		result[row[0]] = row[1]
	
	return result

def spmv_coo_spark_partitioned(spark: SparkSession,
							   coo_matrix: sparse.coo_matrix,
							   x_vector: np.ndarray) -> np.ndarray:
	# Convert COO matrix to DataFrame
	matrix_df = spark.createDataFrame(
		zip(coo_matrix.row, coo_matrix.col, coo_matrix.data),
		schema=["row", "col", "val"]
	)
	
	# Broadcast the vector x
	x_broadcast = spark.sparkContext.broadcast(x_vector)
	
	def process_partition(partition):
		results = []
		for row, col, val in partition:
			results.append((row, val * x_broadcast.value[col]))
		return results

	partial_rdd = matrix_rdd.mapPartitions(process_partition)
	
	# Collect results into a dense array
	result: np.ndarray = np.zeros(coo_matrix.shape[0])
	for row_id, value in result_df.collect():
		result[row_id] = value
	
	return result

def benchmark_spmv_spark(spark: SparkSession,
						 coo_matrix: sparse.coo_matrix,
						 x_vector: np.ndarray,
						 num_iterations=100) -> float:
	
	matrix_rdd: RDD = spark.sparkContext.parallelize(
		zip(coo_matrix.row, coo_matrix.col, coo_matrix.data)
	)
	
	# Broadcast the vector x
	x_broadcast = spark.sparkContext.broadcast(x_vector)
	
	# Warmup
	start_time = time.time()
	partial_rdd = matrix_rdd.map(lambda x: (x[0], x[2] * x_broadcast.value[x[1]]))
	result_rdd: List[Tuple[int, float]] = partial_rdd.reduceByKey(lambda a, b: a + b)\
										  .collect()
	warmup_time = time.time() - start_time
	#print(f"\tWarmup time: {warmup_time * 1000:.4f} ms")

	# Benchmark
	times = []
	for _ in range(num_iterations):
		start_time = time.time()
		partial_rdd = matrix_rdd.map(lambda x: (x[0], x[2] * x_broadcast.value[x[1]]))
		result_rdd: List[Tuple[int, float]] = partial_rdd.reduceByKey(lambda a, b: a + b)\
											  .collect()
		times.append(time.time() - start_time)
	
	avg_time = np.mean(times) * 1000 
	gflops = (2.0 * coo_matrix.nnz / (avg_time / 1000)) / 1e9
	
	# Cleanup
	x_broadcast.unpersist()

	#print("Average time: {avg_time:8.4f} ms")
	
	return avg_time

def benchmark_spmv_spark_row(spark: SparkSession,
							 coo_matrix: sparse.coo_matrix,
							 x_vector: np.ndarray,
							 num_iterations=100) -> float:
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

	#print("Average time: {avg_time:8.4f} ms")
	
	return avg_time

def benchmark_spmv_spark_dataframe(spark: SparkSession,
								 coo_matrix: sparse.coo_matrix,
								 x_vector: np.ndarray,
								 num_iterations=100) -> float:
	# Define explicit schema
	schema = StructType([
		StructField("row", IntegerType(), False),
		StructField("col", IntegerType(), False),
		StructField("val", FloatType(), False)
	])

	# Convert to Python types from numpy
	row_data = [(int(r), int(c), float(v)) for r, c, v in 
				zip(coo_matrix.row, coo_matrix.col, coo_matrix.data)]

	# Create DataFrame with explicit schema
	matrix_df = spark.createDataFrame(row_data, schema=schema)
	
	# Broadcast the vector x
	x_broadcast = spark.sparkContext.broadcast(x_vector)

	# Use a UDF to access broadcast variable
	@udf(FloatType())
	def vector_lookup(col_idx):
		return float(x_broadcast.value[int(col_idx)])
	
	# Warmup
	start_time = time.time()
	partial_df = matrix_df.withColumn("x_val", col("val") * vector_lookup(col("col")))
	result_df = partial_df.groupBy("row").agg({"x_val": "sum"}).select("row", "sum(x_val)")
	warmup_time = time.time() - start_time
	print(f"Warmup time: {warmup_time * 1000:.4f} ms")

	# Benchmark
	times = []
	for _ in range(num_iterations):
		start_time = time.time()
		partial_df = matrix_df.withColumn("x_val", col("val") * vector_lookup(col("col")))
		result_df = partial_df.groupBy("row").agg({"x_val": "sum"}).select("row", "sum(x_val)")
		times.append(time.time() - start_time)
	
	avg_time = np.mean(times) * 1000 
	gflops = (2.0 * coo_matrix.nnz / (avg_time / 1000)) / 1e9
	
	# Cleanup
	x_broadcast.unpersist()

	return avg_time

def test_spmv_spark():	
	spark = SparkSession.builder\
			.appName("SpMV")\
			.getOrCreate()
	

	file = path_utils.get_full_path_from_relative_path("test_matrices/pkustk14.mtx")
	matrix = read_coo_matrix(file)
	
	# Create random vector x
	x = np.random.random(matrix.shape[1])
	
	#row_result: np.ndarray = spmv_coo_spark_row(spark, matrix, x)
	#spark_result: np.ndarray = spmv_coo_spark(spark, matrix, x)
	#sequential_result: np.ndarray = sequential_spmv(matrix, x)
	databse_result: np.ndarray = spmv_coo_spark_dataframe(spark, matrix, x)
	c_result: np.ndarray = spmv_c(matrix, x)
	#assert np.allclose(databse_result, sequential_result), "Results do not match!"
	#assert np.allclose(c_result, sequential_result), "Results do not match!"
	for i in range(len(c_result)):
		if abs(c_result[i] - databse_result[i]) > 0.01:
			print(f"Results do not match at index {i}: {c_result[i]} != {databse_result[i]}")
			break
	print("Results match!")
	
	spark.stop()

def main():
	parser = argparse.ArgumentParser(description='Sparse Matrix-Vector Multiplication using Spark')
	parser.add_argument('matrix_path', type=str, help='Path to the matrix market file (.mtx)')
	args = parser.parse_args()
	#print(f"Matrix path: {args.matrix_path}")

	matrix_file: str = path_utils.get_full_path_from_relative_path(args.matrix_path)
	matrix_file_name = path_utils.get_file_name_without_extension(matrix_file)
	print(f"Matrix: {matrix_file_name}")
	spark = SparkSession.builder\
			.appName("SpMV")\
			.getOrCreate()
	matrix = read_coo_matrix(matrix_file)
	
	# Create random vector x
	x = np.random.random(matrix.shape[1])
	
	# Run benchmark
	#avg_time: float = benchmark_spmv_spark_dataframe(spark, matrix, x, 20)
	#print(f"Average time: {avg_time:.4f} ms")
	sequential_result: np.ndarray = spmv_c(matrix, x)
	database_result: np.ndarray = spmv_coo_spark_dataframe(spark, matrix, x)
	#assert np.allclose(database_result, sequential_result), "Results do not match!"
	for i in range(len(sequential_result)):
		if abs(sequential_result[i] - database_result[i]) > 0.01:
			print(f"Results do not match at index {i}: {sequential_result[i]} != {database_result[i]}")
			break
	print("Results match!")

	spark.stop()

if __name__ == "__main__":
	main()
	#test_spmv_spark()