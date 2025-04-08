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



spmv_c_lib = "libspmv.so"

class ResultsWriter:
	def __init__(self, filename):
		self.filename = filename
		# Open file and clear its contents
		with open(self.filename, 'w') as f:
			f.write(f"SpMV Benchmark Results\n")
			f.write(f"=====================\n\n")
	
	def write(self, message):
		# Write message to file and also print it
		with open(self.filename, 'a') as f:
			f.write(message + "\n")
		print(message)  # Still print to console
	
	def write_performance(self, method_name, time_ms, gflops):
		with open(self.filename, 'a') as f:
			f.write(f"{method_name:<25} {time_ms:>10.4f} ms  {gflops:>8.2f} GFLOP/s\n")
		print(f"{method_name:<25} {time_ms:>10.4f} ms  {gflops:>8.2f} GFLOP/s")


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
	lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), spmv_c_lib)
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

def update_result_array_c(result: np.ndarray, chunk_result: List[Tuple[int, float]]) -> None:
    """Update result array using C implementation for better performance"""
    # Load the shared library
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), spmv_c_lib)
    array_ops_lib = ctypes.CDLL(lib_path)
    
    # Define function argument types
    array_ops_lib.update_array_values.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # array
        ctypes.POINTER(ctypes.c_int),    # indices
        ctypes.POINTER(ctypes.c_float),  # values
        ctypes.c_int                     # count
    ]
    
    # Extract indices and values
    indices = np.array([idx for idx, val in chunk_result], dtype=np.int32)
    values = np.array([val for idx, val in chunk_result], dtype=np.float32)
    count = len(chunk_result)
    
    # Get pointers to arrays
    result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    indices_ptr = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    values_ptr = values.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Call C function
    array_ops_lib.update_array_values(result_ptr, indices_ptr, values_ptr, count)