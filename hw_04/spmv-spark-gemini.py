import sys
import numpy as np
from pyspark import SparkConf, SparkContext
import time
import os

def parse_matrix_market_line(line):
	""" Parses a non-comment line from a Matrix Market file (COO format). """
	parts = line.strip().split()
	if len(parts) == 3:
		# Matrix Market is 1-based, convert to 0-based index
		row = int(parts[0]) - 1
		col = int(parts[1]) - 1
		val = float(parts[2])
		return (row, col, val)
	else:
		return None

def spmv_coo_sequential(matrix_data, vector_x, num_rows):
	"""
	Performs sequential Sparse Matrix-Vector multiplication (y = A*x).

	Args:
		matrix_data: A list of (row, col, value) tuples for non-zero elements.
		vector_x: The input numpy vector x.
		num_rows: The number of rows in the matrix.

	Returns:
		A numpy array representing the resulting vector y.
		Processing time in milliseconds.
	"""
	if not isinstance(vector_x, np.ndarray):
		vector_x = np.array(vector_x)
	y = np.zeros(num_rows, dtype=vector_x.dtype) # Match dtype of x
	start_time = time.time()
	for row, col, val in matrix_data:
		y[row] += val * vector_x[col]
	end_time = time.time()
	processing_time_ms = (end_time - start_time) * 1000
	return y, processing_time_ms

def spmv_coo_spark(sc, num_rows, num_cols, vector_x, matrix_path=None, matrix_rdd=None):
	"""
	Performs Spark Sparse Matrix-Vector multiplication (y = A*x).
	Can take either a matrix_path or a pre-existing matrix_rdd.

	Args:
		sc: The SparkContext.
		num_rows: The number of rows in the matrix.
		num_cols: The number of columns in the matrix.
		vector_x: The input numpy vector x.
		matrix_path: Path to the Matrix Market file (.mtx). (Optional)
		matrix_rdd: An existing RDD of (row, col, val) tuples. (Optional)

	Returns:
		A dictionary representing the resulting vector y {row_index: value}.
		Processing time in milliseconds.
	"""
	if not matrix_path and matrix_rdd is None:
		print("Error (Spark): Need either matrix_path or matrix_rdd.", file=sys.stderr)
		return None, 0
	if matrix_path and matrix_rdd is not None:
		print("Warning (Spark): Both matrix_path and matrix_rdd provided. Using matrix_rdd.", file=sys.stderr)
		matrix_path = None # Prioritize RDD if both given

	start_time = time.time()

	# Check if vector dimension matches matrix columns
	if len(vector_x) != num_cols:
		print(f"Error (Spark): Input vector size ({len(vector_x)}) does not match matrix columns ({num_cols})", file=sys.stderr)
		return None, 0

	# 1. Broadcast the input vector x
	x_bcast = sc.broadcast(vector_x)
	# print("Input vector broadcasted.") # Reduce noise

	# 2. Ensure we have the matrix RDD
	if matrix_rdd is None:
		# Load from path if RDD wasn't provided
		if not os.path.exists(matrix_path):
			print(f"Error (Spark): Matrix file not found at {matrix_path}", file=sys.stderr)
			x_bcast.unpersist()
			return None, 0
		print(f"Spark loading matrix from: {matrix_path}")
		matrix_rdd = sc.textFile(matrix_path) \
						.filter(lambda line: not line.strip().startswith('%')) \
						.zipWithIndex() \
						.filter(lambda line_with_index: line_with_index[1] > 0) \
						.map(lambda line_with_index: line_with_index[0]) \
						.map(parse_matrix_market_line) \
						.filter(lambda x: x is not None)
	# else:
		# print("Spark using pre-parallelized RDD.") # Reduce noise


	# 3. Perform the core SpMV calculation
	partial_results_rdd = matrix_rdd.map(lambda item: (item[0], item[2] * x_bcast.value[item[1]]))

	# 4. Aggregate results by row index
	y_rdd = partial_results_rdd.reduceByKey(lambda a, b: a + b)

	# 5. Collect the result vector y back to the driver
	y_result_map = y_rdd.collectAsMap()
	# print("Spark calculation complete, results collected.") # Reduce noise

	# Cleanup broadcast variable
	x_bcast.unpersist()

	end_time = time.time()
	processing_time_ms = (end_time - start_time) * 1000

	return y_result_map, processing_time_ms


def load_matrix_data(matrix_path):
	""" Reads matrix market file, extracts dimensions and data. """
	num_rows, num_cols, num_nonzeros = 0, 0, 0
	matrix_data = []
	try:
		with open(matrix_path, 'r') as f:
			# Skip comments
			while True:
				line = f.readline()
				if not line: # End of file before finding dimensions
					raise ValueError("Matrix file ended before dimensions line.")
				if line.startswith('%'):
					continue
				# First non-comment line is dimensions
				parts = line.strip().split()
				if len(parts) >= 3:
					num_rows = int(parts[0])
					num_cols = int(parts[1])
					num_nonzeros = int(parts[2])
					print(f"Matrix dimensions: Rows={num_rows}, Cols={num_cols}, NonZeros={num_nonzeros}")
					break
				else:
					raise ValueError(f"Could not parse dimension line: {line.strip()}")

			# Read data lines
			for line in f:
				if line.strip() == "" or line.startswith('%'): # Skip empty/comment lines
					continue
				parsed = parse_matrix_market_line(line)
				if parsed:
					matrix_data.append(parsed)
				else:
					print(f"Warning: Skipping malformed line: {line.strip()}", file=sys.stderr)

		# Optional: Verify non-zero count roughly matches
		if len(matrix_data) != num_nonzeros:
			print(f"Warning: Header reported {num_nonzeros} non-zeros, but {len(matrix_data)} were parsed.", file=sys.stderr)

		return num_rows, num_cols, num_nonzeros, matrix_data

	except FileNotFoundError:
		print(f"Error: Matrix file not found at {matrix_path}", file=sys.stderr)
		return 0, 0, 0, None
	except ValueError as e:
		print(f"Error reading matrix file: {e}", file=sys.stderr)
		return 0, 0, 0, None
	except Exception as e:
		print(f"An unexpected error occurred reading matrix: {e}", file=sys.stderr)
		return 0, 0, 0, None

if __name__ == "__main__":
	if len(sys.argv) != 2:
		print("Usage: spark-submit <this_script.py> <path_to_matrix.mtx>")
		sys.exit(1)

	matrix_file_path = sys.argv[1]

	# --- Load Matrix Data Sequentially (Once) ---
	print(f"Loading matrix data from {matrix_file_path}...")
	num_rows, num_cols, nnz, parsed_matrix_data = load_matrix_data(matrix_file_path)

	if parsed_matrix_data is None or num_rows <= 0 or num_cols <= 0:
		print("Failed to load matrix data. Exiting.", file=sys.stderr)
		sys.exit(1)

	# --- Prepare Input Vector ---
	np.random.seed(42) # for reproducibility
	# Use float64 for potentially better precision in comparison, though C used float
	vector_dtype = np.float64
	x_vector = np.random.rand(num_cols).astype(vector_dtype)
	print(f"Generated random input vector x of size {num_cols} (dtype={vector_dtype})")


	# --- Run Sequential SpMV ---
	print("\nStarting Sequential SpMV calculation...")
	y_sequential, seq_time_ms = spmv_coo_sequential(parsed_matrix_data, x_vector, num_rows)

	if y_sequential is None:
		print("Sequential SpMV failed. Exiting.", file=sys.stderr)
		sys.exit(1)
	print(f"Sequential SpMV Calculation finished in {seq_time_ms:.2f} ms")

	# --- Spark Setup ---
	print("\nInitializing Spark Context...")
	conf = SparkConf().setAppName("Spark SpMV COO Comparison")
	# conf.setMaster("local[*]") # Uncomment for local run without spark-submit
	sc = SparkContext(conf=conf)
	sc.setLogLevel("WARN") # Reduce verbosity
	print(f"Spark Application Initialized. UI at: {sc.uiWebUrl}")

	# --- Run Spark SpMV ---
	print("Starting Spark SpMV calculation...")
	# Parallelize the already parsed data for direct comparison
	matrix_data_rdd = sc.parallelize(parsed_matrix_data)
	# Alternatively, pass the path:
	# y_spark_map, spark_time_ms = spmv_coo_spark(sc, num_rows, num_cols, x_vector, matrix_path=matrix_file_path)
	y_spark_map, spark_time_ms = spmv_coo_spark(sc, num_rows, num_cols, x_vector, matrix_rdd=matrix_data_rdd)

	if y_spark_map is None:
		print("Spark SpMV failed. Exiting.", file=sys.stderr)
		sc.stop()
		sys.exit(1)
	print(f"Spark SpMV Calculation finished in {spark_time_ms:.2f} ms")


	# --- Compare Results ---
	print("\nComparing Sequential and Spark results...")

	# Convert Spark's sparse map result to a dense numpy array for comparison
	y_spark_array = np.zeros(num_rows, dtype=vector_dtype)
	for idx, val in y_spark_map.items():
		if 0 <= idx < num_rows:
			y_spark_array[idx] = val
		else:
			print(f"Warning (Comparison): Spark result index {idx} out of bounds (0-{num_rows-1}).", file=sys.stderr)


	# Use numpy.allclose for robust floating-point comparison
	# Adjust atol (absolute tolerance) and rtol (relative tolerance) if needed
	results_match = np.allclose(y_sequential, y_spark_array, rtol=1e-5, atol=1e-8)

	if results_match:
		print("Results Match!")
	else:
		print("Results DO NOT Match!")
		# Optionally print differences for debugging
		diff = np.abs(y_sequential - y_spark_array)
		max_diff = np.max(diff)
		max_diff_idx = np.argmax(diff)
		print(f"   Max absolute difference: {max_diff:.6e} at index {max_diff_idx}")
		print(f"   Sequential[idx]: {y_sequential[max_diff_idx]:.6e}")
		print(f"   Spark[idx]     : {y_spark_array[max_diff_idx]:.6e}")
		# Print first few elements for visual check
		print("\nFirst 10 elements:")
		print(f"Seq : {y_sequential[:10]}")
		print(f"Spark: {y_spark_array[:10]}")


	# --- Stop Spark ---
	print("\nStopping Spark context.")
	sc.stop()